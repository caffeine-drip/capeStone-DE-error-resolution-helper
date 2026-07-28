"""
rag.py — Retrieval-Augmented Generation for DE troubleshooting.

Pipeline:
  query → embed → [keyword search + vector search] → RRF fusion → LLM answer

Observability:
  * print-based step logging  ([1/5] … [5/5])
  * Logfire spans per pipeline step
  * token usage + estimated cost per query
  * dlt pipeline writing query traces to data/rag_traces.duckdb

Usage:
    from rag import RAG
    r = RAG()
    print(r.query("Spark executor OOM during shuffle"))

    python rag.py "Spark executor OOM during shuffle"
"""

from __future__ import annotations

import os
import time
import uuid
import warnings
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from ingest import build_indices

# ----------------------------------------------------------------- config ----

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11435/v1")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "local")

# RRF constant — 60 is the standard default
RRF_K = 60
TOP_N = 10          # candidates per search leg
TOP_RESULTS = 5     # final docs passed to LLM

KEYWORD_BOOST = {
    "title": 3.0,
    "summary": 2.0,
    "tags_str": 1.5,
    "text": 1.0,
    "component": 0.5,
    "error_type": 0.5,
}

ANSWER_PROMPT = """\
You are a senior data engineering expert. A user has a question about a data engineering error or issue.
Use ONLY the context documents below to answer. Be specific — mention exact config fixes, commands, or
error codes where available. If the context doesn't cover it, say so.

Question: {question}

Context documents:
{context}

Answer:"""


# ------------------------------------------------------------- telemetry ----
#
# TOKENS — set these as Windows environment variables (not in .env):
#   LOGFIRE_TOKEN=<write token from logfire.pydantic.dev>
#   (LOGFIRE_READ_TOKEN only needed for querying traces)
# Get tokens at: https://logfire.pydantic.dev
#
# PowerShell (persistent, user scope):
#   [Environment]::SetEnvironmentVariable("LOGFIRE_TOKEN", "<token>", "User")
#   [Environment]::SetEnvironmentVariable("LOGFIRE_READ_TOKEN", "<token>", "User")
#
# If LOGFIRE_TOKEN is missing, telemetry is silently disabled — the RAG
# pipeline keeps working exactly the same, just without spans.

TRACE_DB_PATH = os.getenv("RAG_TRACE_DB", "data/rag_traces.duckdb")

# Pricing per 1M tokens (USD). Local models are free.
PRICING = {
    "deepseek": {"input": 0.07, "output": 0.28},
    "local": {"input": 0.0, "output": 0.0},
}


def _provider_from_base_url(base_url: str) -> str:
    return "deepseek" if "deepseek" in (base_url or "").lower() else "local"


LLM_PROVIDER = _provider_from_base_url(LLM_BASE_URL)


def estimate_cost(input_tokens: int, output_tokens: int, provider: str = LLM_PROVIDER) -> float:
    """Estimated USD cost for a completion, based on the configured provider."""
    rates = PRICING.get(provider, PRICING["local"])
    return (input_tokens / 1_000_000) * rates["input"] + (
        output_tokens / 1_000_000
    ) * rates["output"]


# ---- logfire (optional) -----------------------------------------------------

_LOGFIRE_ENABLED = False
try:  # pragma: no cover - depends on local env
    import logfire as _logfire

    _token = os.environ.get("LOGFIRE_TOKEN")
    if _token:
        _logfire.configure(
            token=_token,
            service_name="de-error-resolution-rag",
            console=False,
        )
        _LOGFIRE_ENABLED = True
    else:
        warnings.warn(
            "LOGFIRE_TOKEN not set — Logfire telemetry disabled. "
            "Set it as a Windows environment variable to enable tracing.",
            stacklevel=1,
        )
except Exception as exc:  # pragma: no cover
    warnings.warn(f"Logfire unavailable ({exc}) — telemetry disabled.", stacklevel=1)


class _NullSpan:
    """No-op stand-in for a logfire span."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set_attribute(self, *_a, **_k):
        return None

    def set_attributes(self, *_a, **_k):
        return None


def span(name: str, **attrs):
    """logfire.span() if configured, else a no-op context manager."""
    if _LOGFIRE_ENABLED:
        return _logfire.span(name, **attrs)
    return _NullSpan()


def log_info(msg: str, **attrs) -> None:
    if _LOGFIRE_ENABLED:
        try:
            _logfire.info(msg, **attrs)
        except Exception:
            pass


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
        log_info("save_trace_failed", error=str(exc))
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


class RAG:
    def __init__(self):
        with span("build_indices"):
            self.keyword_index, self.vector_store = build_indices()
        self.embed_client = OpenAI(
            api_key="local", base_url=EMBED_BASE_URL, timeout=30, max_retries=0
        )
        # Use DEEPSEEK_API_KEY when pointing at DeepSeek, else "local" for llama.cpp
        llm_api_key = (
            os.environ.get("DEEPSEEK_API_KEY", "local")
            if LLM_PROVIDER == "deepseek"
            else "local"
        )
        self.llm_client = OpenAI(
            api_key=llm_api_key, base_url=LLM_BASE_URL, timeout=120, max_retries=0
        )

    # --------------------------------------------------------- retrieval ----

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
    ) -> tuple[list[dict], dict]:
        """hybrid_search + timing/step metrics (internal)."""
        metrics: dict[str, Any] = {}

        # [1/5] + [3/5] embedding is part of the vector leg; time it separately
        print("[1/5] Embedding query...")
        t0 = time.perf_counter()
        with span("embed_query", query=query):
            query_vec = self._embed(query)
        metrics["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        print("[2/5] Keyword search (minsearch)...")
        t0 = time.perf_counter()
        with span("keyword_search", query=query) as sp:
            kw_results = self.keyword_search(query, filters=filters, n=TOP_N)
            sp.set_attribute("n_results", len(kw_results))
        metrics["keyword_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        print("[3/5] Vector search (cosine similarity)...")
        t0 = time.perf_counter()
        with span("vector_search", query=query) as sp:
            vec_results = self.vector_store.search(query_vec, n=TOP_N)
            sp.set_attribute("n_results", len(vec_results))
        metrics["vector_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        print("[4/5] RRF fusion...")
        with span("rrf_fusion", n_keyword=len(kw_results), n_vector=len(vec_results)) as sp:
            fused_ids = rrf([kw_results, vec_results])[:n]
            id_to_doc: dict[str, dict] = {}
            for doc in kw_results + vec_results:
                id_to_doc.setdefault(doc["id"], doc)
            docs = [id_to_doc[doc_id] for doc_id in fused_ids if doc_id in id_to_doc]
            sp.set_attribute("n_final", len(docs))

        metrics["n_keyword_results"] = len(kw_results)
        metrics["n_vector_results"] = len(vec_results)
        metrics["n_final_docs"] = len(docs)
        return docs, metrics

    # --------------------------------------------------------- generation ----

    def _answer(self, question: str, docs: list[dict]) -> str:
        answer, _ = self._answer_traced(question, docs)
        return answer

    def _answer_traced(self, question: str, docs: list[dict]) -> tuple[str, dict]:
        """LLM call returning (answer, usage/cost metrics)."""
        prompt = ANSWER_PROMPT.format(
            question=question,
            context=format_context(docs),
        )
        print("[5/5] Generating answer (LLM)...")
        t0 = time.perf_counter()
        with span("llm_answer", n_docs=len(docs), model=LLM_MODEL, provider=LLM_PROVIDER) as sp:
            extra = (
                {"extra_body": {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}}}
                if LLM_PROVIDER == "local"
                else {}
            )
            resp = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
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
            cost = estimate_cost(input_tokens, output_tokens)

            sp.set_attribute("input_tokens", input_tokens)
            sp.set_attribute("output_tokens", output_tokens)
            sp.set_attribute("total_tokens", total_tokens)
            sp.set_attribute("estimated_cost_usd", cost)

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

    # --------------------------------------------------------- public API ----

    def query(
        self,
        question: str,
        filters: dict | None = None,
        n_results: int = TOP_RESULTS,
        return_docs: bool = False,
    ) -> str | dict:
        session_id = str(uuid.uuid4())
        started = time.perf_counter()

        with span("rag_query", query=question, session_id=session_id) as root:
            docs, search_metrics = self._hybrid_search_traced(
                question, filters=filters, n=n_results
            )
            answer, llm_metrics = self._answer_traced(question, docs)

            total_ms = round((time.perf_counter() - started) * 1000, 2)
            root.set_attribute("total_ms", total_ms)
            root.set_attribute("n_final_docs", len(docs))
            root.set_attribute("estimated_cost_usd", llm_metrics["estimated_cost_usd"])

        top = docs[:5]
        trace = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": question,
            "n_keyword_results": search_metrics.get("n_keyword_results", 0),
            "n_vector_results": search_metrics.get("n_vector_results", 0),
            "n_final_docs": search_metrics.get("n_final_docs", len(docs)),
            "top_doc_ids": [d.get("id", "") for d in top],
            "top_doc_titles": [d.get("title", "") for d in top],
            "top_components": [d.get("component", "") for d in top],
            "input_tokens": llm_metrics["input_tokens"],
            "output_tokens": llm_metrics["output_tokens"],
            "total_tokens": llm_metrics["total_tokens"],
            "estimated_cost_usd": llm_metrics["estimated_cost_usd"],
            "llm_model": LLM_MODEL,
            "llm_provider": LLM_PROVIDER,
            "embed_ms": search_metrics.get("embed_ms", 0.0),
            "keyword_ms": search_metrics.get("keyword_ms", 0.0),
            "vector_ms": search_metrics.get("vector_ms", 0.0),
            "llm_ms": llm_metrics["llm_ms"],
            "total_ms": total_ms,
            "answer_preview": answer[:200],
        }
        save_trace(trace)
        log_info("rag_query_complete", **{k: trace[k] for k in ("session_id", "total_ms")})

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

    question = " ".join(sys.argv[1:]) or "Spark executor OOM during shuffle — how do I fix it?"
    print(f"Q: {question}\n")
    print(f"Logfire: {'enabled' if _LOGFIRE_ENABLED else 'disabled (no LOGFIRE_TOKEN)'}")
    print(f"LLM: {LLM_MODEL} @ {LLM_BASE_URL} (provider={LLM_PROVIDER})\n")
    r = RAG()
    result = r.query(question, return_docs=True)
    print(f"\nA: {result['answer']}\n")
    print("--- Retrieved docs ---")
    for doc in result["docs"]:
        print(f"  [{doc.get('component')}] {doc.get('title')} — {doc.get('summary', '')[:80]}")
