from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env (LLM_PROVIDER, etc.) before _resolve_eval_llm_config() reads it
# below — without this, LLM_PROVIDER in .env has no effect when these eval
# scripts are run directly (as opposed to inside Docker, where env_file
# already injects it).
load_dotenv()


# Same three-provider convenience switch as my_assistant/rag.py's
# resolve_llm_config(): LLM_PROVIDER=local|deepseek|openai in .env picks the
# right base URL and model automatically for this eval script's own small
# auxiliary LLM calls (LLM-as-judge scoring) — kept separate from whichever
# provider RAG() itself uses to generate answers, but driven by the exact
# same env var for consistency.
_LOCAL_LLM_BASE_URL = "http://localhost:11434/v1"
_LOCAL_LLM_MODEL = "local"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# "deepseek-chat" (the old non-thinking legacy alias) was discontinued
# 2026-07-24 — use the current model name directly.
_DEEPSEEK_MODEL = "deepseek-v4-flash"
_OPENAI_BASE_URL = "https://api.openai.com/v1"
_OPENAI_MODEL = "gpt-4o-mini"

_PROVIDER_API_KEY_ENV = {"deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY"}


def _resolve_eval_llm_config() -> tuple[str, str, str]:
    explicit_url = os.getenv("LLM_BASE_URL")
    if explicit_url:
        model = os.getenv("LLM_MODEL", "local")
        url = explicit_url.lower()
        provider = "deepseek" if "deepseek" in url else "openai" if "openai" in url else "local"
        return explicit_url, model, provider

    provider_choice = os.getenv("LLM_PROVIDER", "local").strip().lower()
    if provider_choice == "deepseek":
        return _DEEPSEEK_BASE_URL, os.getenv("DEEPSEEK_MODEL", _DEEPSEEK_MODEL), "deepseek"
    if provider_choice == "openai":
        return _OPENAI_BASE_URL, os.getenv("OPENAI_MODEL", _OPENAI_MODEL), "openai"
    return (
        os.getenv("LOCAL_LLM_BASE_URL", _LOCAL_LLM_BASE_URL),
        os.getenv("LOCAL_LLM_MODEL", _LOCAL_LLM_MODEL),
        "local",
    )


LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER = _resolve_eval_llm_config()

# The fixed, one-time-generated test-question set that steps 1, 2, and 3 all
# score against. Lives alongside the eval scripts (not in data/) since it's
# test evidence, not part of the knowledge base itself. There is no
# regeneration path anymore — this file was generated once and is reused
# from here on, so results across runs/scripts stay comparable.
EVAL_QUESTIONS_FILE = Path(__file__).resolve().parent / "test_questions_for_eval.json"

CANDIDATES_PER_LEG = 10
REQUEST_TIMEOUT = 180


# --------------------------------------------------------------- clients ----


def make_llm_client() -> OpenAI:
    key_env = _PROVIDER_API_KEY_ENV.get(LLM_PROVIDER)
    api_key = os.environ.get(key_env, "local") if key_env else "local"
    return OpenAI(
        api_key=api_key, base_url=LLM_BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0
    )


def no_think_kwargs() -> dict:
    """Disable Qwen thinking mode so JSON comes back in `content` (local only)."""
    if LLM_PROVIDER != "local":
        return {}
    return {
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}
        }
    }


# ------------------------------------------------------- json extraction ----

try:
    # Reuse the battle-tested extractor (handles <think> blocks, fences, etc.)
    from data_pipeline.enrich_documents import extract_json
except Exception:  # pragma: no cover — fallback keeps this module standalone
    import re

    def extract_json(raw: str) -> dict | None:  # type: ignore[misc]
        if not raw:
            return None
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.I).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------------- io ----


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def rank_of(doc_id: str, results: list[dict], k: int) -> int | None:
    """1-based rank of doc_id within the first k results, else None."""
    for rank, doc in enumerate(results[:k], start=1):
        if doc.get("id") == doc_id:
            return rank
    return None


def target_doc_ids(pair: dict) -> list[str]:
    """The set of document IDs that count as a correct answer for one eval question.

    Almost every question has exactly one correct document ("doc_id"). A few
    (the Spark config-tuning questions merged in from test_questions.md) map
    to a scenario that two near-duplicate documents in the corpus both
    describe — for those, "doc_ids" holds both, and a hit against *either* one
    counts, since both are genuinely correct answers to the question.
    """
    ids = pair.get("doc_ids")
    if isinstance(ids, list) and ids:
        return [str(i) for i in ids]
    single = pair.get("doc_id")
    return [str(single)] if single else []


def best_rank_of(target_ids: list[str], results: list[dict], k: int) -> int | None:
    """Best (lowest) 1-based rank among any of target_ids within the first k results."""
    best: int | None = None
    for tid in target_ids:
        rank = rank_of(tid, results, k)
        if rank is not None and (best is None or rank < best):
            best = rank
    return best


# ------------------------------------------------------ eval questions ----


def load_eval_questions(limit: int | None = None) -> list[dict]:
    """Load the fixed test-question set from evaluation/test_questions_for_eval.json.

    This file was generated once and is reused from here on — there is no
    regeneration path in this module. If you genuinely need a new set (e.g.
    the corpus changed enough to invalidate the old questions), create one by
    hand and save it to EVAL_QUESTIONS_FILE in the same
    [{"question": ..., "doc_id": ...}, ...] shape (or "doc_ids": [...] for a
    question with more than one correct answer).
    """
    if not EVAL_QUESTIONS_FILE.exists():
        raise SystemExit(
            f"{EVAL_QUESTIONS_FILE} not found. This project reuses a fixed, "
            "one-time-generated test-question set — there is no command to "
            "regenerate it. Restore the file or create a new one in the same "
            '[{"question": ..., "doc_id": ...}, ...] shape.'
        )
    pairs = json.loads(EVAL_QUESTIONS_FILE.read_text(encoding="utf-8"))
    pairs = [p for p in pairs if p.get("question") and target_doc_ids(p)]
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise SystemExit("Eval question set is empty — nothing to evaluate against")
    return pairs
