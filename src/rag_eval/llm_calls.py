"""Every LLM call in the evaluation goes through cached_chat():

- Disk cache: the reply is stored under a hash of (provider, model, settings,
  messages). Re-running an evaluation reuses every answer and verdict for free,
  gives identical numbers, and resumes after a crash or a daily rate limit.
  Change the model, a setting or a prompt and the hash changes, so nothing stale
  is reused.
- Pacing: a minimum interval between real API calls per role, to stay under the
  free tiers' per-minute limits (cache hits are never paced).
- Backoff: rate-limit / server errors are retried (llm_utils.call_with_backoff).
"""
from __future__ import annotations

import hashlib
import json
import time

from llama_index.core.llms import ChatMessage, MessageRole

from rag_eval.config import resolve_path
from rag_eval.llm_utils import call_with_backoff
from rag_eval.llms import get_llm

_LLMS: dict[str, object] = {}
_LAST_CALL: dict[str, float] = {}
_ROLES = {"system": MessageRole.SYSTEM, "user": MessageRole.USER}
_INTERVAL_KEYS = {"generator": "generator_min_interval_s", "judge": "judge_min_interval_s"}


def _cache_key(section: dict, messages: list[tuple[str, str]]) -> str:
    material = {
        "provider": section["provider"], "model": section["model"],
        "temperature": section.get("temperature"), "max_tokens": section.get("max_tokens"),
        "extra": section.get("extra", {}), "messages": messages,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def cached_chat(cfg: dict, role: str, messages: list[tuple[str, str]], what: str = "") -> tuple[str, bool, float]:
    """Send [(role, content), ...] to the LLM configured as cfg[role].

    Returns (text, from_cache, api_seconds). api_seconds is the duration of the successful
    API request only (pacing pauses and rate-limit waits excluded). It is stored in the cache,
    so a cached reply still reports how long the model originally took (None for cache
    entries written before this was added).
    """
    section = cfg[role]
    key = _cache_key(section, messages)
    path = resolve_path(cfg, "cache_dir") / "llm" / key[:2] / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["content"], True, data.get("api_s")

    llm_id = json.dumps(section, sort_keys=True)
    if llm_id not in _LLMS:
        _LLMS[llm_id] = get_llm(section)
    llm = _LLMS[llm_id]

    interval = cfg.get("eval", {}).get(_INTERVAL_KEYS.get(role, ""), 0)
    wait = _LAST_CALL.get(role, 0) + interval - time.time()
    if wait > 0:
        time.sleep(wait)

    chat_messages = [ChatMessage(role=_ROLES[r], content=c) for r, c in messages]
    timing: dict[str, float] = {}

    def _call():
        t0 = time.time()
        result = llm.chat(chat_messages)
        timing["api_s"] = time.time() - t0  # only the attempt that succeeded
        return result

    try:
        reply = call_with_backoff(_call, what or role)
    finally:
        _LAST_CALL[role] = time.time()
    content = (reply.message.content or "").strip()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"model": section["model"], "content": content,
                               "api_s": round(timing["api_s"], 3), "created": time.time()}), encoding="utf-8")
    tmp.replace(path)
    return content, False, timing["api_s"]
