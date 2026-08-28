import os
import re
import time
from typing import Dict, List, Optional, Union, Callable, Literal
import logging
import json
from openai import OpenAI, AzureOpenAI
from autogen.agentchat import Agent, UserProxyAgent, ConversableAgent
from termcolor import colored
import Levenshtein

logger = logging.getLogger(__name__)


def _chat_completion(config: dict, messages: list, max_tokens: int = 800):
    """OpenAI-compatible or Azure OpenAI chat.completions."""
    engine = config["model"]
    api_type = str(config.get("api_type", "openai")).lower()
    if api_type == "azure":
        client = AzureOpenAI(
            api_key=config["api_key"],
            azure_endpoint=config["base_url"],
            api_version=config.get("api_version") or "2024-02-15-preview",
        )
    else:
        kw = {"api_key": config["api_key"] or "dummy"}
        if config.get("base_url"):
            kw["base_url"] = config["base_url"]
        client = OpenAI(**kw)
    return client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )


class MedAgent(UserProxyAgent):
    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        llm_config: Optional[Union[Dict, Literal[False]]] = False,
        system_message: Optional[Union[str, List]] = "",
        config_list: Optional[List[Dict]] = None,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
        )
        self.config_list = config_list
        self.question = ""
        self.code = ""
        self.knowledge = ""
        self._last_py_cell_norm = ""
        self._last_py_output = ""
        self._force_terminate_after_exec = False
        self.quiet = False
        self.compress_prompt = False

    def retrieve_knowledge(self, config, query):
        if self.dataset == "mimic_iii":
            from prompts_mimic import RetrKnowledge
        else:
            from prompts_eicu import RetrKnowledge
        patience = 2
        sleep_time = 30
        query_message = RetrKnowledge.format(question=query)
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps people find information."},
            {"role": "user", "content": query_message},
        ]
        while patience > 0:
            patience -= 1
            try:
                response = _chat_completion(config, messages)
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction is not None:
                    return prediction
            except Exception as e:
                print(e)
                err = str(e).lower()
                if "incorrect model" in err or ("400" in err and "model" in err):
                    print(
                        "Hint: retrieve_knowledge and the coding agent use --llm (not --worldmm_llm). "
                        "Set --llm to an id your gateway allows (e.g. gpt-4o-mini), or fix account permissions.",
                        flush=True,
                    )
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "Fail to retrieve related knowledge, please try again later."

    def _nearest_example_indices(self, query: str) -> List[int]:
        """Indices into ``self.memory`` for the ``num_shots`` closest questions (Levenshtein)."""
        if not self.memory:
            return []
        dist = {i: Levenshtein.distance(query, self.memory[i]["question"]) for i in range(len(self.memory))}
        ranked = sorted(dist.items(), key=lambda x: x[1], reverse=False)
        k = min(self.num_shots, len(ranked))
        return [ranked[j][0] for j in range(k)]

    def selected_example_indices(self, query: str) -> List[int]:
        """Same selection as ``retrieve_examples`` (for logging / tests)."""
        return self._nearest_example_indices(query)

    def retrieve_examples(self, query):
        selected_indexes = self._nearest_example_indices(query)
        examples = []
        for i in selected_indexes:
            template = "Question: {}\nKnowledge:\n{}\nSolution:\n{}\n".format(
                self.memory[i]["question"], self.memory[i]["knowledge"], self.memory[i]["code"]
            )
            examples.append(template)
        examples = "\n".join(examples)
        return examples

    def generate_init_message(self, **context):
        self._last_py_cell_norm = ""
        self._last_py_output = ""
        self._force_terminate_after_exec = False
        if self.dataset == "mimic_iii":
            from prompts_mimic import EHRAgent_Message_Prompt
            from question_families import plan_hints_for_family, resolve_question_family
        else:
            from prompts_eicu import EHRAgent_Message_Prompt
            plan_hints_for_family = None
            resolve_question_family = None
        self.question = context["message"]
        family_hint = {}
        reference_python = None
        if resolve_question_family is not None and plan_hints_for_family is not None:
            q_tag, family, entity = resolve_question_family(context["message"])
            family_hint = plan_hints_for_family(q_tag, family, entity, question=context["message"])
            reference_python = family_hint.get("reference_python") if family_hint else None
        if reference_python:
            knowledge = "Use the benchmark family rule below."
        else:
            knowledge = self.retrieve_knowledge(self.config_list[0], context["message"])

        examples = self.retrieve_examples(context["message"])
        if getattr(self, "compress_prompt", False):
            from prompt_compressor import compress_examples, compress_knowledge

            examples = compress_examples(examples, context["message"])
            knowledge = compress_knowledge(knowledge, context["message"])

        if reference_python:
            steps = "\n".join(f"- {s}" for s in family_hint.get("strategy_steps", []))
            knowledge = (
                f"{knowledge}\n\n"
                "Benchmark family rule:\n"
                f"{steps}\n"
                "Use this exact solution pattern unless execution proves it wrong:\n"
                f"{reference_python}"
            )
        self.knowledge = knowledge

        init_message = EHRAgent_Message_Prompt.format(examples=examples, knowledge=knowledge, question=context["message"])
        return init_message

    def send(self, message: Union[Dict, str], recipient: Agent, request_reply: Optional[bool] = None, silent: Optional[bool] = False):
        valid = self._append_oai_message(message, "assistant", recipient, is_sending=True)
        if valid:
            recipient.receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    def initiate_chat(self, recipient: "ConversableAgent", clear_history: Optional[bool] = True, silent: Optional[bool] = False, **context):
        self._prepare_chat(recipient, clear_history)
        self.send(self.generate_init_message(**context), recipient, silent=silent)

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)

    def error_debugger(self, config, code, error_info):
        if self.dataset == "mimic_iii":
            from prompts_mimic import CodeDebugger
        else:
            from prompts_eicu import CodeDebugger
        patience = 2
        sleep_time = 30
        query_message = CodeDebugger.format(question=self.question, code=code, error_info=error_info)
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people debug their code. Only list one most possible reason to the errors.",
            },
            {"role": "user", "content": query_message},
        ]
        while patience > 0:
            patience -= 1
            try:
                response = _chat_completion(config, messages)
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction is not None:
                    return prediction
            except Exception as e:
                print(e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "Fail to diagnose the reasons to the errors."

    @staticmethod
    def _normalize_cell(cell: str) -> str:
        return re.sub(r"\s+", " ", (cell or "").strip())

    @staticmethod
    def _python_answer_looks_final(output: str, code: str) -> bool:
        out = (output or "").strip()
        if not out or out.startswith("Error:") or out.startswith("Error\n"):
            return False
        if len(out) > 400 or out.startswith('{"cell"'):
            return False
        if out.rstrip().endswith("TERMINATE"):
            return True
        c = (code or "").lower()
        if "answer" not in c and "sqlinterpreter" not in c:
            return False
        if out.startswith("[") and out.endswith("]") and len(out) < 120:
            return True
        if "\n" not in out:
            return True
        return False

    def execute_function(self, func_call, **kwargs):
        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)

        is_exec_success = False
        if func is not None:
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = None
                arguments_string = func_call["arguments"].split(': "')[-1]
                arguments_string = arguments_string.split('", ')[0]
                arguments = {"cell": arguments_string}
                content = f"Error: {e}\n There might be compilation errors in the code. Please check the code and try again."

            if arguments is not None:
                cell = arguments.get("cell", "")
                if "\\n" in cell and "\n" not in cell:
                    # Qwen2.5 through vLLM's Hermes parser may double-escape
                    # multiline tool arguments into one literal backslash-n line.
                    cell = re.sub(r"\\+n", "\n", cell)
                if "\\_" in cell:
                    # The same parser occasionally renders escaped newlines as
                    # backslash + underscore before the next identifier.
                    cell = re.sub(r"\\+_", "\n", cell)
                arguments["cell"] = cell
                cell_norm = self._normalize_cell(arguments.get("cell", ""))
                if (
                    func_name == "python"
                    and cell_norm
                    and cell_norm == self._last_py_cell_norm
                    and self._last_py_output
                ):
                    if not getattr(self, "quiet", False):
                        print(colored("\n>>>>>>>> SKIP duplicate python cell (same as last success)", "yellow"), flush=True)
                    return True, {
                        "name": func_name,
                        "role": "function",
                        "content": self._last_py_output,
                    }
                if not getattr(self, "quiet", False):
                    print(colored(f"\n>>>>>>>> EXECUTING FUNCTION {func_name}...", "magenta"), flush=True)
                self.code = arguments["cell"]
                try:
                    content = func(**arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            content = f"Error: Function {func_name} not found."
        if "error" in content or "Error" in content:
            reasons = self.error_debugger(self.config_list[0], self.code, content)
            # Keep debugger short — long rationales blow 8k local context mid-chat.
            content = content + "\nPotential Reasons: " + str(reasons)[:400]
        else:
            if func_name == "python" and is_exec_success:
                out = str(content)
                self._last_py_cell_norm = self._normalize_cell(self.code)
                self._last_py_output = out
                if self._python_answer_looks_final(out, self.code):
                    self._force_terminate_after_exec = True

        content_s = str(content)
        max_tool = int(os.environ.get("EHRAGENT_MAX_TOOL_CHARS", "1400") or 1400)
        if max_tool > 0 and len(content_s) > max_tool:
            content_s = content_s[:max_tool] + "\n# ... tool output truncated for context ..."

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": content_s,
        }

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        super().receive(message, sender, request_reply, silent)
        if getattr(self, "_force_terminate_after_exec", False):
            self._force_terminate_after_exec = False
            self.send("TERMINATE", sender, silent=silent)

    def update_memory(self, num_shots, memory):
        self.num_shots = num_shots
        self.memory = memory

    def register_dataset(self, dataset):
        self.dataset = dataset
