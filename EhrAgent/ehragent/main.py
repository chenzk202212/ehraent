"""EHRMA entrypoint with optional Patient State and Dynamic Memory.

Restored after /home/czk/EhrAgent was deleted. The EHRMA extension includes a
custom WorldMM patch at WorldMM/src/worldmm/memory/ehr.py plus the local EHRSQL
timeline and reporting helpers; these files are not provided by upstream WorldMM.
"""

import os
import sys

# EhrAgent repo root must be first so ``from tools import tabtools`` resolves to
# EhrAgent/tools/, not a PyPI ``tools`` package in the active venv (e.g. WorldMM).
_EHRAGENT_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _EHRAGENT_REPO not in sys.path:
    sys.path.insert(0, _EHRAGENT_REPO)

import json
import random
import numpy as np
import argparse
import autogen
from toolset_high import *
from config import openai_config, llm_config_list
import time


def judge(pred, ans):
    if isinstance(ans, list):
        ans_str = ", ".join(str(x) for x in ans)
    else:
        ans_str = str(ans)
    old_flag = ans_str in pred
    if "True" in pred:
        pred = pred.replace("True", "1")
    else:
        pred = pred.replace("False", "0")
    if ans == "False" or ans == "false":
        ans = "0"
    if ans == "True" or ans == "true":
        ans = "1"
    if ans == "No" or ans == "no":
        ans = "0"
    if ans == "Yes" or ans == "yes":
        ans = "1"
    if ans == "None" or ans == "none":
        ans = "0"
    if isinstance(ans, str) and ", " in ans:
        ans = ans.split(", ")
    if isinstance(ans, str) and len(ans) >= 2 and ans[-2:] == ".0":
        ans = ans[:-2]
    if not isinstance(ans, list):
        ans = [ans]
    new_flag = True
    for i in range(len(ans)):
        if str(ans[i]) not in pred:
            new_flag = False
            break
    return (old_flag or new_flag)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


_LOG_SEP = "\n----------------------------------------------------------\n"


def _chat_logs_joined(thread: list) -> str:
    """Conversation text only (no prepended gold answer — avoids judge false positives)."""
    parts: list[str] = []
    for m in thread:
        if isinstance(m, dict):
            _append_message_to_logs(m, parts)
    return _LOG_SEP.join(parts)


def _log_parts(chat_logs_joined: str) -> list[str]:
    return [p.strip() for p in chat_logs_joined.split(_LOG_SEP) if p.strip()]


def _segment_is_code_cell(segment: str) -> bool:
    s = segment.strip()
    return s.startswith('{"cell"') or '"cell"' in s[:300]


def _python_output_looks_final(output: str, code: str = "") -> bool:
    """Scalar / short tool return after code sets ``answer`` (route, cost, 0/1, etc.)."""
    out = (output or "").strip()
    if not out or out.startswith("Error:") or out.startswith("Error\n"):
        return False
    if _segment_is_code_cell(out) or len(out) > 400:
        return False
    c = (code or "").lower()
    if "answer" not in c and "answer =" not in c.replace(" ", ""):
        return False
    if "\n" not in out:
        return True
    if out.startswith("[") and out.endswith("]") and len(out) < 120:
        return True
    return False


def _assistant_terminated(thread: list, chat_logs_joined: str = "") -> bool:
    """
    True when the run finished with TERMINATE after a code execution.
    Autogen may omit role on assistant messages or use tool/function roles for python output.
    Also accept a successful short python return after code that sets ``answer`` (no TERMINATE).
    """
    skip_roles = {"user", "function", "tool"}
    for m in reversed(thread):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").lower()
        if role in skip_roles:
            continue
        if m.get("tool_calls") or m.get("function_call"):
            continue
        content = m.get("content")
        if content is None:
            continue
        text = content if isinstance(content, str) else str(content)
        if text.rstrip().endswith("TERMINATE"):
            return True

    parts = _log_parts(chat_logs_joined)
    if not parts:
        return False
    if parts[-1].rstrip().endswith("TERMINATE"):
        return True
    if parts[-1] == "TERMINATE":
        if len(parts) < 2:
            return False
        pre = parts[-2]
        if pre.startswith("Error:") or pre.startswith("Error\n"):
            return False
        if _segment_is_code_cell(pre):
            return len(parts) >= 3
        for p in reversed(parts[:-2]):
            if _segment_is_code_cell(p):
                return True
        return False

    last = parts[-1]
    for p in reversed(parts[:-1]):
        if _segment_is_code_cell(p):
            cell = p
            if '"cell"' in p:
                try:
                    parsed = json.loads(p) if p.strip().startswith("{") else None
                    if isinstance(parsed, dict) and parsed.get("cell"):
                        cell = parsed["cell"]
                except json.JSONDecodeError:
                    pass
            if _python_output_looks_final(last, cell):
                return True
            return False
    return False


def _prediction_region_for_judge(chat_logs_joined: str) -> str:
    """
    EHRAgent few-shot prompt contains many literal ``Solution:`` lines; do not use rfind('Solution:').
    Use the last python tool return through the end of the chat (do not cut at TERMINATE inside the
    initial prompt / memory plan — that would include the gold answer line and inflate PASS rate).
    """
    marker = "***** Response from calling function (python) *****"
    i = chat_logs_joined.rfind(marker)
    if i >= 0:
        return chat_logs_joined[i:]
    parts = _log_parts(chat_logs_joined)
    if len(parts) >= 2 and parts[-1] == "TERMINATE":
        start = len(parts) - 2
        for idx in range(len(parts) - 2, -1, -1):
            p = parts[idx]
            if (
                not _segment_is_code_cell(p)
                and p != "TERMINATE"
                and not p.startswith("Error:")
                and not p.startswith("Potential Reasons:")
            ):
                start = idx
                break
        else:
            for idx, p in enumerate(parts):
                if _segment_is_code_cell(p):
                    start = idx
        return _LOG_SEP.join(parts[start:])
    return chat_logs_joined


def _append_message_to_logs(msg: dict, logs_string: list) -> None:
    """Collect text / code from autogen message (legacy function_call or tool_calls)."""
    if not isinstance(msg, dict):
        return
    c = msg.get("content")
    if c is not None and str(c).strip():
        logs_string.append(str(c))
    fc = msg.get("function_call")
    if fc:
        arg = fc.get("arguments")
        if isinstance(arg, dict) and "cell" in arg:
            logs_string.append(arg["cell"])
        else:
            logs_string.append(str(arg))
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        if fn.get("name") != "python":
            continue
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "cell" in parsed:
                    logs_string.append(parsed["cell"])
                else:
                    logs_string.append(raw)
            except json.JSONDecodeError:
                logs_string.append(raw)
        elif isinstance(raw, dict) and "cell" in raw:
            logs_string.append(raw["cell"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--num_questions", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mimic_iii")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--logs_path", type=str, default="./logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_id", type=str, default="521fd2885f51641a963f8d3e")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Keep full per-question logs on disk but suppress verbose chat/code transcripts in the console.",
    )
    parser.add_argument(
        "--compress_prompt",
        action="store_true",
        help="Structure-aware compression for examples and memory blocks before sending prompts to the LLM.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="Run one ad-hoc natural-language question instead of loading --data_path.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read one ad-hoc question from stdin instead of loading --data_path.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for one pasted ad-hoc question. End with a /done line or Ctrl-D.",
    )
    parser.add_argument(
        "--paste",
        action="store_true",
        help="Alias for --interactive.",
    )
    parser.add_argument(
        "--question_id",
        type=str,
        default="",
        help="Run only the row with this JSON id (use with --no_shuffle for stable indexing).",
    )
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--num_shots", type=int, default=4)
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Keep JSON row order (no random.shuffle). Default matches upstream: shuffle for variety.",
    )
    parser.add_argument(
        "--memory_trace",
        action="store_true",
        help="Log long-term memory: pool size, seed vs success-added count, Levenshtein-selected example indices each question.",
    )
    parser.add_argument(
        "--ltm_disable",
        action="store_true",
        help="Paper ablation: do not append successful runs to long_term_memory (pool stays seed-only).",
    )
    parser.add_argument(
        "--use_worldmm_ehr",
        action="store_true",
        help="Prepend WorldMM EHRMWorldMemory retrieval. With mimic_iii, also builds timelines from "
        "EHRAGENT_DATA_ROOT/mimic_iii CSVs using question-visible patient anchors when "
        "--worldmm_timeline_dir has no file; gold SQL is never consulted.",
    )
    parser.add_argument(
        "--memory_agent",
        action="store_true",
        help="Memory agent mode: structured WorldMM + task memory drive a planner policy before coding. "
        "Implies --use_worldmm_ehr. Persists task memory under logs/.task_memory.json.",
    )
    parser.add_argument(
        "--no_worldmm_context",
        action="store_true",
        help="Ablation: MemoryAgent with Memory Plan but no WorldMM belief context.",
    )
    parser.add_argument(
        "--planner_heuristic_only",
        action="store_true",
        help="Memory agent: skip planner LLM call (no extra API cost); use rule-based table/strategy plan.",
    )
    parser.add_argument(
        "--harness_retry_on_fail",
        action="store_true",
        help="Failure-triggered harness: on FAIL/unfinished, arm task-memory constraints and retry the same question once.",
    )
    parser.add_argument(
        "--worldmm_root",
        default="",
        help="WorldMM repo root (contains src/worldmm). Default: EhrAgent/WorldMM.",
    )
    parser.add_argument(
        "--worldmm_timeline_dir",
        default="",
        help="Optional dir of pre-built timeline JSON (_mimic_hadm_<HADM>.json / {id}.json). "
        "If no match, mimic_iii falls back to CSV timeline under logs/.worldmm_ehrsql_cache/.",
    )
    parser.add_argument(
        "--worldmm_device",
        default=os.environ.get("WORLDMM_DEVICE", "cpu"),
        help="WorldMM EmbeddingModel device (cuda | cpu).",
    )
    parser.add_argument(
        "--worldmm_llm",
        default=os.environ.get("WORLDMM_LLM", "").strip(),
        help="Model for WorldMM EHRMWorldMemory retrieval only (many API calls per question). "
        "Default: same as --llm. On AIHubMix, if gpt-4-0613 fails here, set e.g. gpt-4o-mini or WORLDMM_LLM.",
    )
    parser.add_argument(
        "--show_llm_endpoint",
        action="store_true",
        help="Print resolved model, base_url, api_type (redacted key), then exit. Use to verify AIHubMix env.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    adhoc_question = args.question.strip() or " ".join(args.prompt).strip()
    should_read_stdin = args.stdin or (
        not adhoc_question
        and not args.data_path
        and not args.interactive
        and not args.paste
        and not args.show_llm_endpoint
        and not sys.stdin.isatty()
    )
    if should_read_stdin:
        stdin_question = sys.stdin.read().strip()
        if stdin_question:
            adhoc_question = stdin_question
    auto_interactive = (
        not adhoc_question
        and not args.data_path
        and not should_read_stdin
        and not args.show_llm_endpoint
    )
    if args.interactive or args.paste or auto_interactive:
        print("Paste one question. Finish with a line containing only /done, or press Ctrl-D:", flush=True)
        lines = []
        try:
            while True:
                line = input()
                if line.strip() == "/done":
                    break
                lines.append(line)
        except EOFError:
            pass
        pasted_question = "\n".join(lines).strip()
        if pasted_question:
            adhoc_question = pasted_question

    cfg0 = openai_config(args.llm)
    if not cfg0.get("api_key"):
        from config import _EHRAGENT_DIR, _REPO_ROOT

        print(
            "MedAgent needs OPENAI_API_KEY (and usually OPENAI_BASE_URL for third-party gateways).\n"
            "  export OPENAI_API_KEY='sk-...'   # or: export AIHUBMIX_API_KEY='...'\n"
            "  export OPENAI_BASE_URL='https://api.openai.com/v1'   # AIHubMix: https://aihubmix.com/v1\n"
            "  export EHRAGENT_DATA_ROOT=/path/to/ehrsql-ehragent\n"
            "Or create a file (not committed to git) with those lines:\n"
            f"  {_EHRAGENT_DIR / '.env'}\n"
            f"  {_REPO_ROOT / '.env'}\n"
            "Then re-run. Check: python main.py --show_llm_endpoint --llm gpt-4o-mini",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.show_llm_endpoint:
        safe = {
            "model": cfg0.get("model"),
            "base_url": cfg0.get("base_url"),
            "api_type": cfg0.get("api_type"),
            "api_key_set": bool(cfg0.get("api_key")),
        }
        print(json.dumps(safe, indent=2), flush=True)
        sys.exit(0)

    if not adhoc_question and (not args.data_path or not os.path.isfile(args.data_path)):
        print(
            "Pass --data_path /path/to/benchmark.json (list of {id, template, answer}), "
            "or use --question '...', --stdin, or --interactive for one pasted question.",
            file=sys.stderr,
        )
        sys.exit(2)
    os.makedirs(args.logs_path, exist_ok=True)

    if args.dataset == "mimic_iii":
        from tools import tabtools as _tabtools

        root = os.environ.get("EHRAGENT_DATA_ROOT", "").strip()
        mimic = _tabtools._mimic_iii_csv_dir()
        if not root or root.startswith("<"):
            print(
                "Set EHRAGENT_DATA_ROOT to your ehrsql-ehragent root (contains mimic_iii/ADMISSIONS.csv).\n"
                "  export EHRAGENT_DATA_ROOT=/home/czk/EhrAgent/ehrsql-ehragent\n"
                "Or add it to ehragent/.env",
                file=sys.stderr,
            )
            sys.exit(2)
        if not os.path.isfile(os.path.join(mimic, "ADMISSIONS.csv")):
            print(f"MIMIC CSVs not found under {mimic}. Check EHRAGENT_DATA_ROOT={root!r}.", file=sys.stderr)
            sys.exit(2)
        try:
            _tabtools._mimic_iii_db_path()
        except FileNotFoundError as e:
            print(f"Warning: {e} — SQLInterpreter will fail; LoadDB/FilterDB still use CSV.", file=sys.stderr)

    set_seed(args.seed)
    if args.dataset == 'mimic_iii':
        from prompts_mimic import EHRAgent_4Shots_Knowledge
    else:
        from prompts_eicu import EHRAgent_4Shots_Knowledge

    config_list = [openai_config(args.llm)]
    llm_config = llm_config_list(args.seed, config_list)
    # Local 8k models: fewer tool rounds → less mid-chat context blowup.
    _local = any(
        "127.0.0.1" in str(c.get("base_url", "")) or "localhost" in str(c.get("base_url", ""))
        for c in config_list
    )
    max_auto_reply = int(os.environ.get("EHRAGENT_MAX_AUTO_REPLY", "7" if _local else "10") or 10)

    _default_wm = os.path.abspath(os.path.join(_EHRAGENT_REPO, "WorldMM"))
    worldmm_root = (args.worldmm_root or "").strip() or (
        _default_wm if os.path.isdir(os.path.join(_default_wm, "src", "worldmm")) else ""
    )

    use_worldmm = args.use_worldmm_ehr or args.memory_agent
    if args.memory_agent and not args.use_worldmm_ehr:
        print("[MemoryAgent] enabling WorldMM backend (--use_worldmm_ehr)", flush=True)

    if use_worldmm:
        if not worldmm_root:
            print(
                f"Pass --worldmm_root /path/to/WorldMM or ensure {_default_wm}/src/worldmm exists.",
                file=sys.stderr,
            )
            sys.exit(2)
        from worldmm_bridge import ensure_worldmm_on_path

        ensure_worldmm_on_path(worldmm_root)
        w_base = (cfg0.get("base_url") or "").rstrip("/")
        wm_llm = (args.worldmm_llm or "").strip() or args.llm
        if wm_llm != args.llm:
            print(
                f"[WorldMM] retrieval LLM={wm_llm!r} (EhrAgent chat/knowledge uses --llm {args.llm!r}).",
                flush=True,
            )
        proxy_kw = dict(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=max_auto_reply,
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,
            },
            config_list=config_list,
            worldmm_llm_name=wm_llm,
            worldmm_base_url=w_base,
            worldmm_device=args.worldmm_device,
        )
        if args.memory_agent:
            from medagent_memory_agent import MedAgentMemoryAgent

            task_mem_path = os.path.join(os.path.abspath(args.logs_path), ".task_memory.json")
            user_proxy = MedAgentMemoryAgent(
                task_memory_path=task_mem_path,
                planner_heuristic_only=args.planner_heuristic_only,
                **proxy_kw,
            )
            user_proxy.quiet = args.quiet
            user_proxy.compress_prompt = args.compress_prompt
            user_proxy.set_memory_trace(args.memory_trace)
            if getattr(args, "no_worldmm_context", False):
                user_proxy._no_worldmm_context = True
                print("[MemoryAgent] ablation: WorldMM context disabled", flush=True)
            print(f"[MemoryAgent] task memory: {task_mem_path}", flush=True)
            if args.harness_retry_on_fail:
                print("[MemoryAgent] failure-triggered harness retry ON", flush=True)
        else:
            from medagent_worldmm import MedAgentWorldMM

            user_proxy = MedAgentWorldMM(**proxy_kw)
            user_proxy.quiet = args.quiet
            user_proxy.compress_prompt = args.compress_prompt
    else:
        from medagent import MedAgent

        user_proxy = MedAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=max_auto_reply,
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,
            },
            config_list=config_list,
        )
        user_proxy.quiet = args.quiet
        user_proxy.compress_prompt = args.compress_prompt

    # register the functions
    user_proxy.register_function(
        function_map={
            "python": run_code
        }
    )

    user_proxy.register_dataset(args.dataset)

    adhoc_mode = bool(adhoc_question)
    if adhoc_mode:
        contents = [
            {
                "id": "adhoc",
                "template": adhoc_question,
                "answer": None,
            }
        ]
        args.no_shuffle = True
        args.num_questions = 1
        args.start_id = 0
        args.debug = False
        args.question_id = ""
    else:
        file_path = args.data_path
        # read from json file
        with open(file_path, 'r') as f:
            contents = json.load(f)

    if not args.no_shuffle:
        random.shuffle(contents)
    file_path = "{}/{}/".format(args.logs_path, args.num_shots) + "{id}.txt"
    os.makedirs(os.path.join(args.logs_path, str(args.num_shots)), exist_ok=True)

    start_time = time.time()
    if args.num_questions == -1:
        args.num_questions = len(contents)

    if args.question_id:
        run_indices = [i for i, row in enumerate(contents) if row.get("id") == args.question_id]
        if not run_indices:
            print(f"question_id not found in dataset: {args.question_id}", file=sys.stderr)
            sys.exit(1)
        if len(run_indices) > 1:
            print(f"warning: duplicate question_id {args.question_id!r}, running first match", flush=True)
    else:
        run_indices = list(range(args.start_id, args.num_questions))
    long_term_memory = []
    n_ok, n_fail, n_unfinished, n_total = 0, 0, 0, 0
    init_memory = EHRAgent_4Shots_Knowledge
    init_memory = init_memory.split('\n\n')
    for i in range(len(init_memory)):
        item = init_memory[i]
        item = item.split('Question:')[-1]
        question = item.split('\nKnowledge:\n')[0]
        item = item.split('\nKnowledge:\n')[-1]
        knowledge = item.split('\nSolution:')[0]
        code = item.split('\nSolution:')[-1]
        new_item = {"question": question, "knowledge": knowledge, "code": code}
        long_term_memory.append(new_item)

    mem_seed_count = len(long_term_memory)
    ltm_success_added = 0

    resolve_timeline_path = None
    worldmm_ehrsql_cache = ""
    if use_worldmm:
        from worldmm_bridge import resolve_timeline_path

        worldmm_ehrsql_cache = os.path.join(os.path.abspath(args.logs_path), ".worldmm_ehrsql_cache")

    for i in run_indices:
        if args.debug and contents[i]['id'] != args.debug_id:
            continue
        n_total += 1
        question = contents[i]['template']
        gold = contents[i].get('answer')
        try:
            if resolve_timeline_path is not None:
                tdir = args.worldmm_timeline_dir.strip() or None
                tp = resolve_timeline_path(contents[i], tdir)
                if not tp and args.dataset == "mimic_iii" and worldmm_ehrsql_cache:
                    from ehrsql_worldmm_timeline import timeline_path_for_benchmark_row

                    tp = timeline_path_for_benchmark_row(
                        contents[i],
                        cache_dir=worldmm_ehrsql_cache,
                    )
                user_proxy.set_worldmm_timeline(tp)
                if not tp:
                    print(
                        f"[WorldMM] no timeline for id={contents[i].get('id')} "
                        f"(optional: --worldmm_timeline_dir; otherwise the question must expose an "
                        f"unambiguous patient/admission anchor and EHRAGENT_DATA_ROOT must contain mimic_iii/*.csv).",
                        flush=True,
                    )
            user_proxy.update_memory(args.num_shots, long_term_memory)
            if args.memory_trace:
                pick = user_proxy.selected_example_indices(question)
                from_success = len(long_term_memory) - mem_seed_count
                hits_ltm = sum(1 for j in pick if j >= mem_seed_count)
                print(
                    f"[LTM] pool={len(long_term_memory)} seed={mem_seed_count} "
                    f"from_prior_success={from_success} num_shots={args.num_shots} "
                    f"picked_indices={pick} (includes {hits_ltm} non-seed)",
                    flush=True,
                )
            # Fresh assistant each item so chat_messages do not accumulate past questions.
            chatbot = autogen.agentchat.AssistantAgent(
                name=f"chatbot_{i}",
                system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done. Save the answers to the questions in the variable 'answer'. Please only generate the code.",
                llm_config=llm_config,
            )
            user_proxy.initiate_chat(
                chatbot,
                message=question,
                q_tag=contents[i].get("q_tag") or contents[i].get("tag"),
                value=contents[i].get("value"),
                silent=args.quiet,
            )

            logs_string = []
            logs_string.append(str(question))
            logs_string.append(str(gold))
            thread = user_proxy.chat_messages.get(chatbot) or []
            for m in thread:
                if isinstance(m, dict):
                    _append_message_to_logs(m, logs_string)
        except Exception as e:
            # Never abort the full benchmark on a single-item LLM/context failure.
            print(f"[EhrAgent] item {i} chat error (continue): {e}", flush=True)
            logs_string = [str(e)]
            thread = []
        print(f"\n--- item {i + 1}/{args.num_questions} id={contents[i].get('id')} ---", flush=True)
        if not args.quiet:
            print(logs_string, flush=True)
        file_directory = file_path.format(id=contents[i]['id'])
        if gold is None:
            logs_string.append("Ground-Truth Answer ---> <none: ad-hoc question>")
        else:
            gold_line = ", ".join(str(x) for x in gold) if isinstance(gold, list) else str(gold)
            logs_string.append("Ground-Truth Answer ---> " + gold_line)
        with open(file_directory, 'w', encoding="utf-8") as f:
            f.write('\n----------------------------------------------------------\n'.join(logs_string))
        logs_joined = '\n----------------------------------------------------------\n'.join(logs_string)
        chat_logs_joined = _chat_logs_joined(thread)
        terminated = _assistant_terminated(thread, chat_logs_joined)
        first_terminated = terminated
        if gold is None:
            result = bool(terminated)
            if terminated:
                n_ok += 1
            else:
                n_unfinished += 1
        elif not terminated:
            n_unfinished += 1
            result = False
        else:
            prediction = _prediction_region_for_judge(chat_logs_joined)
            result = judge(prediction, gold)
            if result:
                n_ok += 1
            else:
                n_fail += 1

        # Failure-triggered harness: FAIL/unfinished → targeted code+error repair retry.
        if (
            args.harness_retry_on_fail
            and args.memory_agent
            and gold is not None
            and not result
            and hasattr(user_proxy, "arm_failure_harness")
        ):
            err = getattr(user_proxy, "_last_error", "") or ""
            if not err:
                for line in reversed(logs_string):
                    low = str(line).lower()
                    if "error" in low or "traceback" in low or "exception" in low:
                        err = str(line)[:500]
                        break
            code0 = getattr(user_proxy, "code", "") or ""
            user_proxy.arm_failure_harness(last_error=err, code=code0)
            try:
                chatbot_r = autogen.agentchat.AssistantAgent(
                    name=f"chatbot_{i}_retry",
                    system_message=(
                        "You are repairing a FAILED EHR coding attempt. "
                        "Use only the provided python tool. "
                        "PATCH the broken code shown in the user message using the stated error. "
                        "Do not ignore the broken code. "
                        "Save the final result in variable answer and reply TERMINATE when done."
                    ),
                    llm_config=llm_config,
                )
                user_proxy.initiate_chat(
                    chatbot_r,
                    message=question,
                    q_tag=contents[i].get("q_tag") or contents[i].get("tag"),
                    value=contents[i].get("value"),
                    silent=args.quiet,
                )
                thread = user_proxy.chat_messages.get(chatbot_r) or []
                logs_string.append("===== FAILURE HARNESS RETRY =====")
                for m in thread:
                    if isinstance(m, dict):
                        _append_message_to_logs(m, logs_string)
                chat_logs_joined = _chat_logs_joined(thread)
                terminated = _assistant_terminated(thread, chat_logs_joined)
                if not terminated:
                    retry_result = False
                else:
                    prediction = _prediction_region_for_judge(chat_logs_joined)
                    retry_result = judge(prediction, gold)
                print(
                    f"[MemoryAgent] failure-harness retry judge={'PASS' if retry_result else 'FAIL'} "
                    f"TERMINATE={terminated} cat={getattr(user_proxy, '_failure_category', '')}",
                    flush=True,
                )
                # Replace first-attempt counters with retry outcome.
                if first_terminated:
                    n_fail = max(0, n_fail - 1)
                else:
                    n_unfinished = max(0, n_unfinished - 1)
                if not terminated:
                    n_unfinished += 1
                    result = False
                elif retry_result:
                    n_ok += 1
                    result = True
                else:
                    n_fail += 1
                    result = False
            except Exception as e:
                logs_string.append(f"Failure harness retry error: {e}")
                print(f"[MemoryAgent] failure-harness retry error: {e}", flush=True)
            finally:
                if hasattr(user_proxy, "clear_failure_harness"):
                    user_proxy.clear_failure_harness()
            # Rewrite per-question log including retry transcript.
            with open(file_directory, "w", encoding="utf-8") as f:
                f.write("\n----------------------------------------------------------\n".join(logs_string))

        if args.memory_agent and hasattr(user_proxy, "trace_is_valid"):
            execution_valid = bool(user_proxy.trace_is_valid(terminated))
        else:
            execution_valid = bool(terminated and getattr(user_proxy, "code", ""))

        # EHRMA online LTM: prefer execution evidence. In harness mode, also keep gold-PASS
        # solutions so later questions / outer evolve can reuse verified successes.
        ltm_write_valid = execution_valid if args.memory_agent else bool(result)
        if args.harness_retry_on_fail and result:
            ltm_write_valid = True
        if ltm_write_valid and not args.ltm_disable:
            code = (user_proxy.code or "").strip()
            if "LoadDB" in code or "SQLInterpreter" in code:
                new_item = {
                    "question": question,
                    "knowledge": user_proxy.knowledge,
                    "code": code,
                }
                long_term_memory.append(new_item)
                ltm_success_added += 1
                if args.memory_trace:
                    print(
                        f"[LTM] append execution-verified id={contents[i].get('id')} "
                        f"-> pool={len(long_term_memory)}",
                        flush=True,
                    )
        elif args.memory_trace and args.ltm_disable:
            print("[LTM] skip append (--ltm_disable)", flush=True)
        if args.memory_agent and hasattr(user_proxy, "finish_question"):
            # Harness mode: gold PASS also writes skills/experiences for later questions.
            mem_success = bool(execution_valid) or (bool(args.harness_retry_on_fail) and bool(result))
            user_proxy.finish_question(question, mem_success)
            if args.memory_trace and mem_success and not execution_valid and result:
                print("[MemoryAgent] recorded gold-PASS success into task memory", flush=True)
        elif use_worldmm and hasattr(user_proxy, "_ehrmm") and user_proxy._ehrmm is not None:
            user_proxy._ehrmm.update_from_experience(
                question=question, success=execution_valid, code=getattr(user_proxy, "code", "")
            )
        print(
            (
                f"judge={'PASS' if result else 'FAIL'}  (substring match vs gold). "
                if gold is not None
                else "judge=SKIP  (no gold answer for ad-hoc question). "
            )
            + f"TERMINATE in log: {terminated}",
            flush=True,
        )
    end_time = time.time()
    log_dir = os.path.join(args.logs_path, str(args.num_shots))
    stats = {
        "total_num": n_total,
        "correct": n_ok,
        "unfinished": n_unfinished,
        "incorrect": n_fail,
    }
    sr = 100.0 * n_ok / n_total if n_total else 0.0
    cr = 100.0 * (n_ok + n_fail) / n_total if n_total else 0.0
    print("")
    print(f"Time elapsed: {end_time - start_time:.1f}s")
    print(json.dumps(stats, indent=4))
    print(f"SR (success rate): {sr:.2f}%  CR (completion rate): {cr:.2f}%")
    print(f"Per-question logs: {log_dir}/<id>.txt")
    print(
        f"LTM summary: seed_entries={mem_seed_count} success_appends={ltm_success_added} "
        f"final_pool={len(long_term_memory)} ltm_growth={'off' if args.ltm_disable else 'on'}",
        flush=True,
    )

if __name__ == "__main__":
    main()
