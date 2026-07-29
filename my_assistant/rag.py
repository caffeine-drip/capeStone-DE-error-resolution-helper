"""
rag.py — Retrieval-Augmented Generation for DE troubleshooting.

Pipeline:
  query → embed → [keyword search + vector search] → RRF fusion → LLM answer

Observability:
  * print-based step logging  ([1/7] … [7/7])
  * token usage + estimated cost per query
  * dlt pipeline writing query traces to data/rag_traces.duckdb

Usage:
    from my_assistant.rag import RAG
    r = RAG()
    print(r.query("Spark executor OOM during shuffle"))

    python my_assistant/rag.py "Spark executor OOM during shuffle"
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

# Lives inside the my_assistant/ package but repo root needs to be on
# sys.path for `from my_assistant.ingest import ...` to resolve when this
# file is run directly (e.g. `uv run python my_assistant/rag.py ...`)
# rather than imported as part of the my_assistant package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from my_assistant.ingest import build_indices

# ----------------------------------------------------------------- config ----

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11435/v1")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "local")

# RRF constant — 60 is the standard default
RRF_K = 60
TOP_N = 10          # candidates per search leg
TOP_RESULTS = 5     # final docs passed to LLM

# ------------------------------------------------------ out-of-scope gate ----
# If retrieval can't find anything genuinely relevant, skip the LLM call
# entirely and return this fixed message instead of letting the model try to
# answer from its own training knowledge (which is exactly what this project
# is trying to avoid — see README §1, "General-purpose LLMs").
OUT_OF_SCOPE_MESSAGE = "Can't be answered using the current playbook."

# Cosine similarity is the only retrieval signal here that's an absolute
# relevance measure rather than a rank (RRF's 1/(k+rank) score reflects
# "ranked higher than the alternatives in this result set", not "is actually
# relevant" — it can't tell a great match from the least-bad of five weak
# ones). This threshold is a reasonable starting point for normalized
# Qwen3-Embedding vectors, not a tuned value — sanity-check it against a
# handful of real in-scope vs. clearly-unrelated questions for your corpus
# and adjust via the MIN_RELEVANCE_SCORE env var if needed.
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))

KEYWORD_BOOST = {
    "title": 1.0,
    "summary": 2.0,
    "tags_str": 1.5,
    "text": 1.0,
    "component": 1.0,
    "error_type": 0.25,
}

# Winner of evaluation/3_llm_prompt_judge_evaluation.py's 4-persona LLM-as-judge
# comparison (2026-07-30): this structured prompt beat the other 3 personas
# (SME-concise, DE-with-examples, error-resolution-terse) on groundedness
# (4.97 vs 4.93/4.87/4.60) and tied for the top relevance score (5.00), at a
# token cost similar to the next-best persona. See
# evaluation/results_3_llm_prompt_judge_evaluation.json for the full comparison.
ANSWER_PROMPT = """\
You are a senior data engineering expert helping a colleague debug a production issue.
Use ONLY the context documents below. Never invent config keys, commands, or error codes
that do not appear in the context. If the context does not cover the question, say so
explicitly instead of guessing.

Structure your answer exactly like this:

**Likely cause**
One or two sentences naming the most probable root cause.

**Troubleshooting steps**
1. ... (each step is a concrete action: a command to run, a config to change,
   a log or metric to check — include exact parameter names and values from the context)
2. ...
3. ...

**How to verify the fix**
One or two sentences on what a healthy result looks like.

**Caveats**
Anything the context does not cover, or trade-offs of the fix. Write "None" if there are none.

Question: {question}

Context documents:
{context}

Answer:"""

VERIFY_PROMPT = """\
You are a strict technical reviewer checking an answer before it's shown to a user.

Original question:
{question}

Context documents the answer was supposed to be grounded in:
{context}

Draft answer:
{answer}

Check the draft for:
1. Internal consistency — does every config value / flag mentioned in the prose also appear in any
   final command shown? Does every flag in a final command match what the prose actually recommended?
2. Completeness against the context — does the answer address the most likely root causes visible in
   the context documents (e.g. skew, AQE, partitioning), or does it jump to one fix while ignoring a
   more relevant one that the context supports?
3. Unsupported claims — anything stated as fact that the context documents don't actually support.

Respond with ONLY JSON, no markdown fences:
{{"consistent": true/false, "issues": ["short specific issue", ...]}}

If there are no real issues, return {{"consistent": true, "issues": []}}."""

REPAIR_PROMPT = """\
You are a senior data engineering expert. Revise the draft answer below to fix the specific issues
listed, while keeping everything that was already correct. Stay grounded in the context documents.
Output only the corrected answer — no preamble, no explanation of what changed.

Original question:
{question}

Context documents:
{context}

Draft answer:
{answer}

Issues to fix:
{issues}

Corrected answer:"""


# ------------------------------------------------------------- telemetry ----

TRACE_DB_PATH = os.getenv("RAG_TRACE_DB", "data/rag_traces.duckdb")

# Pricing per 1M tokens (USD). Local models are free.
# OpenAI default model is gpt-4o-mini ($0.15 in / $0.60 out per 1M tokens, 2026 pricing).
PRICING = {
    "deepseek": {"input": 0.07, "output": 0.28},
    "openai": {"input": 0.15, "output": 0.60},
    "local": {"input": 0.0, "output": 0.0},
}


def _provider_from_base_url(base_url: str) -> str:
    url = (base_url or "").lower()
    if "deepseek" in url:
        return "deepseek"
    if "openai" in url:
        return "openai"
    return "local"


LLM_PROVIDER = _provider_from_base_url(LLM_BASE_URL)


def estimate_cost(input_tokens: int, output_tokens: int, provider: str = LLM_PROVIDER) -> float:
    """Estimated USD cost for a completion, based on the configured provider."""
    rates = PRICING.get(provider, PRICING["local"])
    return (input_tokens / 1_000_000) * rates["input"] + (
        output_tokens / 1_000_000
    ) * rates["output"]


# ---- dlt trace pipeline -----------------------------------------------------

try:  # pragma: no cover - depends on local env
    import dlt

    _DLT_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    dlt = None  # type: ignore[assignment]
    _DLT_AVAILABLE = False
    warnings.warn(f"dlt unavailable ({exc}) — trace persistence disabled.", stacklevel=1)


if _DLT_AVAILABLE:

    @dlt.resource(name="rag_traces", write_disposition="append")
    def traces_resource(trace: dict):
        """dlt resource yielding a single RAG query trace row."""
        yield trace

else:  # pragma: no cover

    def traces_resource(trace: dict):
        yield trace


def save_trace(trace: dict) -> bool:
    """
    Best-effort: persist a query trace to data/rag_traces.duckdb via dlt.

    Never raises — a telemetry failure must not break a user query.
    """
    if not _DLT_AVAILABLE:
        return False
    try:
        db_dir = os.path.dirname(TRACE_DB_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        pipeline = dlt.pipeline(
            pipeline_name="rag_telemetry",
            destination=dlt.destinations.duckdb(TRACE_DB_PATH),
            dataset_name="rag",
        )
        pipeline.run(traces_resource(trace))
        return True
    except Exception as exc:
        warnings.warn(f"save_trace failed (non-fatal): {exc}", stacklevel=1)
        return False


# ----------------------------------------------------------------- helpers ----


def rrf(ranked_lists: list[list[Any]], k: int = RRF_K) -> list[Any]:
    """Reciprocal Rank Fusion over multiple ranked lists of doc IDs."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = item if isinstance(item, str) else item.get("id", "")
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda x: -scores[x])


def extract_json_object(raw: str) -> dict | None:
    """Pull a JSON object out of a model response (balanced-brace scan,
    same approach used in enrich_documents.py)."""
    import json
    import re

    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    if "<think>" in text:
        text = text.split("</think>")[-1].strip() if "</think>" in text else ""

    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escape = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        return parsed if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        return None
    return None


def format_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        section = f"[{i}] {doc.get('title', 'Untitled')}"
        if doc.get("summary"):
            section += f"\nSummary: {doc['summary']}"
        body = doc.get("text") or doc.get("problem") or ""
        if body:
            section += f"\n{body[:800]}"
        resolution = doc.get("resolution") or ""
        if resolution:
            section += f"\nResolution: {resolution[:400]}"
        parts.append(section)
    return "\n\n---\n\n".join(parts)


# ------------------------------------------------------------------- RAG -----


LOCAL_LLM_BASE_URL = "http://localhost:11434/v1"
LOCAL_LLM_MODEL = "local"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# "deepseek-chat" (the old non-thinking legacy alias) was discontinued
# 2026-07-24 — use the current model name directly.
DEEPSEEK_MODEL = "deepseek-v4-flash"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"

# Env var name -> where its API key comes from. Real key VALUES are never
# read from .env — only from the actual process/OS environment (Windows user
# env var, or passed through by `docker-compose --env-file` / `environment:`
# at container start). .env only ever holds the provider CHOICE (LLM_PROVIDER)
# and non-secret settings like OPENAI_MODEL.
_PROVIDER_API_KEY_ENV = {"deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY"}


def resolve_llm_config() -> tuple[str, str, str]:
    """
    Resolve (base_url, model, provider) for the default RAG() instance.

    Priority:
      1. Explicit LLM_BASE_URL (+ optional LLM_MODEL) — advanced/manual override,
         e.g. pointing at a non-standard local port.
      2. LLM_PROVIDER=local|deepseek|openai — the convenience switch meant to be
         set in .env / docker-compose. Picks the right base URL and a sensible
         default model automatically.
      3. Falls back to local llama.cpp defaults if neither is set.
    """
    explicit_url = os.getenv("LLM_BASE_URL")
    if explicit_url:
        return explicit_url, os.getenv("LLM_MODEL", "local"), _provider_from_base_url(explicit_url)

    provider_choice = os.getenv("LLM_PROVIDER", "local").strip().lower()
    if provider_choice == "deepseek":
        return DEEPSEEK_BASE_URL, os.getenv("DEEPSEEK_MODEL", DEEPSEEK_MODEL), "deepseek"
    if provider_choice == "openai":
        return OPENAI_BASE_URL, os.getenv("OPENAI_MODEL", OPENAI_MODEL), "openai"
    return (
        os.getenv("LOCAL_LLM_BASE_URL", LOCAL_LLM_BASE_URL),
        os.getenv("LOCAL_LLM_MODEL", LOCAL_LLM_MODEL),
        "local",
    )


class RAG:
    def __init__(
        self,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
    ):
        """
        llm_base_url / llm_model let a caller (e.g. the Streamlit app) pick the
        provider per-instance. If not given, resolved from the environment via
        resolve_llm_config() — i.e. LLM_PROVIDER=local|deepseek|openai in .env.
        """
        if llm_base_url:
            self.llm_base_url = llm_base_url
            self.llm_model = llm_model or LLM_MODEL
            self.llm_provider = _provider_from_base_url(self.llm_base_url)
        else:
            self.llm_base_url, self.llm_model, self.llm_provider = resolve_llm_config()

        self.keyword_index, self.vector_store = build_indices()
        self.embed_client = OpenAI(
            api_key="local", base_url=EMBED_BASE_URL, timeout=30, max_retries=0
        )
        # Real API key values only ever come from the process environment —
        # a Windows user env var when run directly, or passed through by
        # docker-compose at container start. Never read from .env/.env.example.
        key_env_var = _PROVIDER_API_KEY_ENV.get(self.llm_provider)
        llm_api_key = os.environ.get(key_env_var, "local") if key_env_var else "local"
        if key_env_var and llm_api_key == "local":
            warnings.warn(
                f"LLM_PROVIDER={self.llm_provider} but {key_env_var} is not set in the "
                f"environment — requests to {self.llm_base_url} will fail. Set it as a "
                f"real OS/user environment variable (never in .env).",
                stacklevel=1,
            )
        self.llm_client = OpenAI(
            api_key=llm_api_key, base_url=self.llm_base_url, timeout=120, max_retries=0
        )

        self._embed_health_cache: tuple[float, bool] | None = None  # (checked_at, available)

    # --------------------------------------------------------- retrieval ----

    def embeddings_available(self, force_recheck: bool = False, cache_seconds: float = 15.0) -> bool:
        """
        Quick health probe for the embedding server (port 11435 by default).
        Cached briefly so "auto" retrieval mode doesn't add a network round-trip
        to every single query — a dead/live server rarely flips within 15s.
        """
        now = time.time()
        if not force_recheck and self._embed_health_cache is not None:
            checked_at, available = self._embed_health_cache
            if now - checked_at < cache_seconds:
                return available

        try:
            self.embed_client.embeddings.create(model="local", input="ping", timeout=3)
            available = True
        except Exception:  # noqa: BLE001 — any failure means "not available"
            available = False

        self._embed_health_cache = (now, available)
        return available

    def _embed(self, text: str) -> list[float]:
        resp = self.embed_client.embeddings.create(model="local", input=text[:4000])
        return resp.data[0].embedding

    def keyword_search(
        self,
        query: str,
        filters: dict | None = None,
        n: int = TOP_N,
    ) -> list[dict]:
        return self.keyword_index.search(
            query,
            filter_dict=filters or {},
            boost_dict=KEYWORD_BOOST,
            num_results=n,
        )

    def vector_search(self, query: str, n: int = TOP_N) -> list[dict]:
        vec = self._embed(query)
        return self.vector_store.search(vec, n=n)

    def hybrid_search(
        self,
        query: str,
        filters: dict | None = None,
        n: int = TOP_RESULTS,
    ) -> list[dict]:
        docs, _ = self._hybrid_search_traced(query, filters=filters, n=n)
        return docs

    def _hybrid_search_traced(
        self,
        query: str,
        filters: dict | None = None,
        n: int = TOP_RESULTS,
        retrieval_mode: str = "auto",
    ) -> tuple[list[dict], dict]:
        """Retrieval + timing/step metrics (internal).

        retrieval_mode:
          "auto"    — (default) probe the embedding server; use hybrid if it's
                      reachable, fall back to keyword-only if it isn't. This is
                      what makes the whole pipeline work even when nobody
                      remembered to start llama-server on :11435.
          "hybrid"  — keyword + vector + RRF fusion, no fallback (errors if the
                      embedding server is down).
          "keyword" — minsearch only, forced. No embedding model/server ever
                      contacted, even if one happens to be running.
          "vector"  — embeddings only, no keyword leg, no fallback.
        """
        resolved_mode = retrieval_mode
        if retrieval_mode == "auto":
            if self.embeddings_available():
                resolved_mode = "hybrid"
            else:
                resolved_mode = "keyword"
                print("[auto] Embedding server unreachable — falling back to keyword-only retrieval")
        retrieval_mode = resolved_mode

        metrics: dict[str, Any] = {
            "embed_ms": 0.0, "keyword_ms": 0.0, "vector_ms": 0.0,
            "n_keyword_results": 0, "n_vector_results": 0,
            "retrieval_mode": retrieval_mode,
            "top_vector_score": None,  # cosine similarity of the best vector
                                       # match, when available — used by the
                                       # out-of-scope gate in query().
        }
        query_vec = None
        kw_results: list[dict] = []
        vec_results: list[dict] = []

        needs_embedding = retrieval_mode in ("hybrid", "vector")
        if needs_embedding:
            print("[1/7] Embedding query...")
            t0 = time.perf_counter()
            query_vec = self._embed(query)
            metrics["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        else:
            print("[1/7] Embedding skipped (keyword-only mode — no embedding model needed)")

        if retrieval_mode in ("hybrid", "keyword"):
            print("[2/7] Keyword search (minsearch)...")
            t0 = time.perf_counter()
            kw_results = self.keyword_search(query, filters=filters, n=TOP_N)
            metrics["keyword_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        else:
            print("[2/7] Keyword search skipped (vector-only mode)")

        if needs_embedding:
            print("[3/7] Vector search (cosine similarity)...")
            t0 = time.perf_counter()
            vec_results = self.vector_store.search(query_vec, n=TOP_N)
            metrics["vector_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            if vec_results:
                metrics["top_vector_score"] = max(
                    float(d.get("_score", 0.0)) for d in vec_results
                )
        else:
            print("[3/7] Vector search skipped")

        if retrieval_mode == "keyword":
            docs = kw_results[:n]
            metrics["n_keyword_results"] = len(kw_results)
            metrics["n_final_docs"] = len(docs)
            print("[4/7] RRF fusion skipped (single retrieval leg)")
            return docs, metrics
        if retrieval_mode == "vector":
            docs = vec_results[:n]
            metrics["n_vector_results"] = len(vec_results)
            metrics["n_final_docs"] = len(docs)
            print("[4/7] RRF fusion skipped (single retrieval leg)")
            return docs, metrics

        print("[4/7] RRF fusion...")
        fused_ids = rrf([kw_results, vec_results])[:n]
        id_to_doc: dict[str, dict] = {}
        for doc in kw_results + vec_results:
            id_to_doc.setdefault(doc["id"], doc)
        docs = [id_to_doc[doc_id] for doc_id in fused_ids if doc_id in id_to_doc]

        metrics["n_keyword_results"] = len(kw_results)
        metrics["n_vector_results"] = len(vec_results)
        metrics["n_final_docs"] = len(docs)
        return docs, metrics

    # --------------------------------------------------------- generation ----

    @staticmethod
    def _is_out_of_scope(search_metrics: dict) -> bool:
        """
        Decide whether retrieval found anything worth answering from, without
        ever calling the LLM. Two independent, deliberately conservative
        signals — either is sufficient on its own:

          - hybrid/vector mode: the best cosine-similarity match is below
            MIN_RELEVANCE_SCORE. This is the only genuinely calibratable
            relevance signal available (RRF's fused score is rank-based, not
            an absolute measure, so it can't tell "great match" from "the
            least-bad of five weak ones").
          - keyword-only mode (or vector unavailable): keyword search
            returned zero hits at all. No score calibration needed here —
            zero results really does mean zero results.
        """
        mode = search_metrics.get("retrieval_mode")
        top_score = search_metrics.get("top_vector_score")
        if mode in ("hybrid", "vector") and top_score is not None:
            return top_score < MIN_RELEVANCE_SCORE
        if mode == "keyword":
            return search_metrics.get("n_keyword_results", 0) == 0
        return False

    def _answer(self, question: str, docs: list[dict]) -> str:
        answer, _ = self._answer_traced(question, docs)
        return answer

    def _answer_traced(self, question: str, docs: list[dict]) -> tuple[str, dict]:
        """LLM call returning (answer, usage/cost metrics)."""
        prompt = ANSWER_PROMPT.format(
            question=question,
            context=format_context(docs),
        )
        print("[5/7] Generating answer (LLM)...")
        t0 = time.perf_counter()
        extra = (
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}}}
            if self.llm_provider == "local"
            else {}
        )
        resp = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
            **extra,
        )
        answer = (resp.choices[0].message.content or "").strip()

        usage = getattr(resp, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or (
            input_tokens + output_tokens
        )
        cost = estimate_cost(input_tokens, output_tokens, provider=self.llm_provider)

        llm_ms = round((time.perf_counter() - t0) * 1000, 2)
        metrics = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "llm_ms": llm_ms,
        }
        print(
            f"      Tokens: {input_tokens} in / {output_tokens} out "
            f"| Cost: ${cost:.4f} | Time: {llm_ms / 1000:.1f}s"
        )
        return answer, metrics

    def _llm_call(self, prompt: str, max_tokens: int, temperature: float = 0.1) -> tuple[str, dict]:
        """Shared low-level LLM call used by verify/repair — same token/cost accounting as _answer_traced."""
        extra = (
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}}}
            if self.llm_provider == "local"
            else {}
        )
        resp = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0) or (input_tokens + output_tokens)
        cost = estimate_cost(input_tokens, output_tokens, provider=self.llm_provider)
        return text, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost,
        }

    def _verify_traced(self, question: str, docs: list[dict], answer: str) -> tuple[dict, dict]:
        """Tool-call-style self-check: does the draft answer hold up against
        itself and the question? Returns ({"consistent": bool, "issues": [...]}, metrics)."""
        print("[6/7] Self-checking answer...")
        t0 = time.perf_counter()
        prompt = VERIFY_PROMPT.format(question=question, context=format_context(docs), answer=answer)
        raw, usage = self._llm_call(prompt, max_tokens=400)
        verdict = extract_json_object(raw) or {"consistent": True, "issues": []}
        verdict["consistent"] = bool(verdict.get("consistent", True))
        verdict["issues"] = list(verdict.get("issues") or [])
        check_ms = round((time.perf_counter() - t0) * 1000, 2)
        usage["check_ms"] = check_ms
        if verdict["issues"]:
            print(f"      Found {len(verdict['issues'])} issue(s): {verdict['issues']}")
        else:
            print("      No issues found.")
        return verdict, usage

    def _repair_traced(self, question: str, docs: list[dict], answer: str, issues: list[str]) -> tuple[str, dict]:
        """One repair pass addressing the issues raised by _verify_traced."""
        print("[7/7] Repairing answer based on self-check...")
        t0 = time.perf_counter()
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        prompt = REPAIR_PROMPT.format(
            question=question, context=format_context(docs), answer=answer, issues=issues_text
        )
        revised, usage = self._llm_call(prompt, max_tokens=1200, temperature=0.2)
        usage["repair_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return revised or answer, usage

    # --------------------------------------------------------- public API ----

    def query(
        self,
        question: str,
        filters: dict | None = None,
        n_results: int = TOP_RESULTS,
        return_docs: bool = False,
        self_check: bool = True,
        retrieval_mode: str = "auto",
    ) -> str | dict:
        """
        retrieval_mode: "auto" (default — uses hybrid if the embedding server
        is reachable, otherwise falls back to keyword-only automatically),
        "hybrid" (force, errors if embeddings are down), "keyword" (force,
        never touches the embedding model), or "vector" (force, embeddings only).
        """
        session_id = str(uuid.uuid4())
        started = time.perf_counter()

        docs, search_metrics = self._hybrid_search_traced(
            question, filters=filters, n=n_results, retrieval_mode=retrieval_mode
        )

        out_of_scope = self._is_out_of_scope(search_metrics)
        check_flagged = False
        check_issues: list[str] = []
        repaired = False
        check_tokens = 0
        check_cost = 0.0
        check_ms = 0.0
        repair_ms = 0.0

        if out_of_scope:
            # Skip the LLM entirely — nothing relevant was retrieved, so
            # there's nothing to ground an answer in. Returning a fixed
            # string here (rather than letting the model try) is the
            # whole point: it can't fall back on its own training
            # knowledge for something our corpus doesn't cover.
            top_score = search_metrics.get("top_vector_score")
            reason = (
                f"top vector score {top_score:.3f} < {MIN_RELEVANCE_SCORE}"
                if top_score is not None
                else "keyword search returned 0 results"
            )
            print(f"[5/7] Skipping answer generation — out of scope ({reason})")
            print("[6/7] Self-check skipped (out of scope)")
            print("[7/7] Repair skipped (out of scope)")
            answer = OUT_OF_SCOPE_MESSAGE
            llm_metrics = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "llm_ms": 0.0,
            }
        else:
            answer, llm_metrics = self._answer_traced(question, docs)

            # Self-check tool call: verify the draft answer against itself
            # and the question before returning it. One repair pass if
            # it's flagged.
            if self_check:
                verdict, check_usage = self._verify_traced(question, docs, answer)
                check_flagged = not verdict["consistent"]
                check_issues = verdict["issues"]
                check_tokens += check_usage["total_tokens"]
                check_cost += check_usage["estimated_cost_usd"]
                check_ms = check_usage.get("check_ms", 0.0)

                if check_flagged and check_issues:
                    revised, repair_usage = self._repair_traced(question, docs, answer, check_issues)
                    answer = revised
                    repaired = True
                    check_tokens += repair_usage["total_tokens"]
                    check_cost += repair_usage["estimated_cost_usd"]
                    repair_ms = repair_usage.get("repair_ms", 0.0)
            else:
                print("[6/7] Self-check skipped")
                print("[7/7] Repair skipped")

        llm_metrics["total_tokens"] += check_tokens
        llm_metrics["estimated_cost_usd"] += check_cost

        total_ms = round((time.perf_counter() - started) * 1000, 2)

        top = docs[:5]
        trace = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": question,
            "n_keyword_results": search_metrics.get("n_keyword_results", 0),
            "n_vector_results": search_metrics.get("n_vector_results", 0),
            "n_final_docs": search_metrics.get("n_final_docs", len(docs)),
            "retrieval_mode": search_metrics.get("retrieval_mode", "hybrid"),
            "top_vector_score": search_metrics.get("top_vector_score"),
            "out_of_scope": out_of_scope,
            "top_doc_ids": [d.get("id", "") for d in top],
            "top_doc_titles": [d.get("title", "") for d in top],
            "top_components": [d.get("component", "") for d in top],
            "input_tokens": llm_metrics["input_tokens"],
            "output_tokens": llm_metrics["output_tokens"],
            "total_tokens": llm_metrics["total_tokens"],
            "estimated_cost_usd": llm_metrics["estimated_cost_usd"],
            "llm_model": self.llm_model,
            "llm_provider": self.llm_provider,
            "embed_ms": search_metrics.get("embed_ms", 0.0),
            "keyword_ms": search_metrics.get("keyword_ms", 0.0),
            "vector_ms": search_metrics.get("vector_ms", 0.0),
            "llm_ms": llm_metrics["llm_ms"],
            "self_check_ms": check_ms,
            "repair_ms": repair_ms,
            "self_check_flagged": check_flagged,
            "self_check_issues": check_issues,
            "repaired": repaired,
            "total_ms": total_ms,
            "answer_preview": answer[:200],
        }
        save_trace(trace)

        print(
            f"Done in {total_ms / 1000:.1f}s "
            f"| {trace['n_final_docs']} docs "
            f"| ${trace['estimated_cost_usd']:.4f}"
        )

        if return_docs:
            return {"answer": answer, "docs": docs, "trace": trace}
        return answer


# --------------------------------------------------------------- CLI demo ----

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    self_check = "--no-check" not in args
    use_deepseek = "--deepseek" in args
    use_openai = "--openai" in args

    # Default: auto — use embeddings if the embedding server responds, else
    # fall back to keyword-only automatically. Force with an explicit flag.
    retrieval_mode = "auto"
    if "--use-embeddings" in args:
        retrieval_mode = "hybrid"  # force on, error out if the server is down
    elif "--no-embeddings" in args or "--keyword-only" in args:
        retrieval_mode = "keyword"  # force off, never touches the embedding model
    elif "--vector-only" in args:
        retrieval_mode = "vector"

    flags = (
        "--no-check", "--deepseek", "--openai",
        "--use-embeddings", "--no-embeddings", "--keyword-only", "--vector-only",
    )
    args = [a for a in args if a not in flags]

    question = " ".join(args) or "Spark executor OOM during shuffle — how do I fix it?"

    if use_deepseek and use_openai:
        print("Pass only one of --deepseek / --openai")
        sys.exit(1)
    if use_deepseek:
        r = RAG(llm_base_url=DEEPSEEK_BASE_URL, llm_model=DEEPSEEK_MODEL)
    elif use_openai:
        r = RAG(llm_base_url=OPENAI_BASE_URL, llm_model=OPENAI_MODEL)
    else:
        # No explicit flag — follows LLM_PROVIDER from .env (local/deepseek/openai).
        r = RAG()

    print(f"Q: {question}\n")
    print(f"LLM: {r.llm_model} @ {r.llm_base_url} (provider={r.llm_provider})")
    print(f"Self-check: {'enabled' if self_check else 'disabled (--no-check)'}")
    print(f"Retrieval mode: {retrieval_mode}"
          + (" (probing embedding server...)" if retrieval_mode == "auto" else "")
          + "\n")
    result = r.query(
        question, return_docs=True, self_check=self_check, retrieval_mode=retrieval_mode
    )
    print(f"(actually used: {result['trace'].get('retrieval_mode', retrieval_mode)})")
    print(f"\nA: {result['answer']}\n")
    if result["trace"].get("repaired"):
        print(f"(answer was revised after self-check flagged: {result['trace']['self_check_issues']})\n")
    print("--- Retrieved docs ---")
    for doc in result["docs"]:
        print(f"  [{doc.get('component')}] {doc.get('title')} — {doc.get('summary', '')[:80]}")
