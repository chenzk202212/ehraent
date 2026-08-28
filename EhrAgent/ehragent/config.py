"""LLM config for pyautogen + MedAgent (OpenAI-compatible or Azure OpenAI)."""

import os
import sys
from pathlib import Path

_EHRAGENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EHRAGENT_DIR.parent


def _load_dotenv_files() -> None:
    """Load KEY=VALUE lines from .env (does not override variables already in the environment)."""
    candidates = [
        _EHRAGENT_DIR / ".env",
        _REPO_ROOT / ".env",
        Path(os.environ.get("EHRAGENT_DOTENV", "")).expanduser(),
    ]
    for path in candidates:
        if not path or not path.is_file():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except OSError:
            pass
        break


_load_dotenv_files()


def _resolve_openai_credentials() -> tuple[str, str]:
    """
    Return (api_key, base_url). Accepts AIHubMix-style env names as aliases.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = os.environ.get("AIHUBMIX_API_KEY", "").strip()
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)

    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not base_url:
        base_url = os.environ.get("AIHUBMIX_BASE_URL", "").strip()
    if not base_url and api_key and os.environ.get("AIHUBMIX_API_KEY", "").strip():
        base_url = "https://aihubmix.com/v1"
    if not base_url:
        base_url = "https://api.openai.com/v1"
    base_url = base_url.rstrip("/")
    os.environ.setdefault("OPENAI_BASE_URL", base_url)
    return api_key, base_url


def openai_config(model: str) -> dict:
    """
    Build one config dict for autogen ``config_list`` and MedAgent.retrieve_knowledge.

    Environment:
      EHRAGENT_API_TYPE   openai (default) | azure
      OPENAI_API_KEY      (or AIHUBMIX_API_KEY — copied to OPENAI_API_KEY automatically)
      OPENAI_BASE_URL     OpenAI-compatible base URL, default https://api.openai.com/v1
                          (AIHubMix: https://aihubmix.com/v1, or AIHUBMIX_BASE_URL)
      OPENAI_API_VERSION  Azure only, default 2024-02-15-preview
      OPENAI_MODEL        Used when --llm is missing or still a placeholder
    """
    raw = os.environ.get("EHRAGENT_API_TYPE", "openai").strip().lower()
    is_azure = raw in ("azure", "az")
    bu_env = os.environ.get("OPENAI_BASE_URL", "").strip()
    if is_azure and "aihubmix" in bu_env.lower():
        print(
            "EhrAgent: EHRAGENT_API_TYPE is 'azure' but OPENAI_BASE_URL looks like AIHubMix. "
            "AIHubMix is OpenAI-compatible: run `unset EHRAGENT_API_TYPE` or "
            "`export EHRAGENT_API_TYPE=openai`, then retry.",
            file=sys.stderr,
            flush=True,
        )

    m = (model or "").strip()
    if not m or m.startswith("<"):
        m = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()

    api_key, base_url = _resolve_openai_credentials()
    if is_azure:
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or base_url
        api_version = os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview").strip()
        api_type = "azure"
        return {
            "model": m,
            "api_key": api_key,
            "base_url": base_url,
            "api_version": api_version,
            "api_type": api_type,
        }

    if "aihubmix.com" in base_url.lower() and not base_url.endswith("/v1"):
        print(
            f"EhrAgent: AIHubMix expects OPENAI_BASE_URL ending with /v1 (e.g. https://aihubmix.com/v1); "
            f"currently: {base_url!r}.",
            file=sys.stderr,
            flush=True,
        )
    # Newer pyautogen passes this dict to OpenAI(); do not set api_version (Azure-only).
    return {
        "model": m,
        "api_key": api_key,
        "base_url": base_url,
            }


def llm_config_list(seed, config_list):
    python_function = {
        "name": "python",
        "description": "run the entire code and return the execution result. Only generate the code.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell": {
                    "type": "string",
                    "description": "Valid Python code to execute.",
                }
            },
            "required": ["cell"],
        },
    }
    llm_config_list = {
        "config_list": config_list,
        "timeout": 600,
        "cache_seed": None,
        "temperature": 0,
    }
    is_local_vllm = any(
        "127.0.0.1" in str(config.get("base_url", ""))
        or "localhost" in str(config.get("base_url", ""))
        for config in config_list
    )
    if is_local_vllm:
        # Leave headroom under 8k ctx (prompt + completion must fit together).
        llm_config_list["max_tokens"] = int(os.environ.get("EHRAGENT_MAX_TOKENS", "512") or 512)
        llm_config_list["tools"] = [{"type": "function", "function": python_function}]
        llm_config_list["tool_choice"] = {"type": "function", "function": {"name": "python"}}
    else:
        llm_config_list["functions"] = [python_function]
    return llm_config_list
