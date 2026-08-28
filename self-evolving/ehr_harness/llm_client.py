"""OpenAI-compatible chat client for harness meta-planning (no weight updates)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_dotenv() -> None:
    roots = [
        Path(os.environ.get("EHRAGENT_ROOT", "/home/czk/EhrAgent")) / "ehragent" / ".env",
        Path(os.environ.get("EHRAGENT_ROOT", "/home/czk/EhrAgent")) / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for path in roots:
        if not path.is_file():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'").strip('"')
                if k and k not in os.environ:
                    os.environ[k] = v
        except OSError:
            pass
        break


_load_dotenv()


def resolve_endpoint(model: str) -> Dict[str, str]:
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("AIHUBMIX_API_KEY")
        or ""
    ).strip()
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("AIHUBMIX_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    return {"api_key": api_key, "base_url": base_url, "model": model}


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    timeout_s: int = 120,
) -> str:
    ep = resolve_endpoint(model)
    if not ep["api_key"]:
        raise RuntimeError(
            "Meta-planner needs OPENAI_API_KEY (or AIHUBMIX_API_KEY). "
            "Put it in EhrAgent/ehragent/.env or export it."
        )
    url = ep["base_url"] + "/chat/completions"
    body = {
        "model": ep["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ep['api_key']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
    return (payload["choices"][0]["message"]["content"] or "").strip()


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None
