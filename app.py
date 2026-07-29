"""
app.py — Streamlit UI for the DE Error-Resolution RAG system.

Two pages (sidebar radio):
  * Chat / Search    — ask a question, get a grounded answer + retrieved docs + feedback
  * Monitoring       — dashboards over data/rag_traces.duckdb (dlt-loaded traces + feedback)

Run:
    streamlit run app.py

The RAG pipeline reads data/ with *relative* paths (see my_assistant/ingest.py), so this
module chdir()s to its own directory on import to make `streamlit run` location-independent.
Core pipeline code (rag.py, ingest.py, kb_ingest.py) lives in my_assistant/; app.py is the
only file meant to be run directly outside Docker.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# --------------------------------------------------------------- bootstrap ---

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
try:
    os.chdir(APP_DIR)
except OSError:
    pass

# Load .env (LLM_PROVIDER, etc.) when running outside Docker. In Docker,
# docker-compose's env_file already injects these into the process
# environment, so this is a no-op there — but it's what makes .env do
# anything at all when running `streamlit run app.py` directly.
load_dotenv()

DATA_DIR = APP_DIR / "data"
DOCS_FILE = DATA_DIR / "enriched_documents.json"
TRACE_DB_PATH = Path(os.getenv("RAG_TRACE_DB", str(DATA_DIR / "rag_traces.duckdb")))

LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11435/v1")
DEEPSEEK_DISPLAY_URL = "https://api.deepseek.com/v1 (deepseek-v4-flash)"
OPENAI_DISPLAY_URL = "https://api.openai.com/v1 (gpt-4o-mini)"

# Consistent provider colors used across every Monitoring chart, so local
# (free, on-GPU), DeepSeek (paid API), and OpenAI (paid API) usage are
# visually distinguishable at a glance.
PROVIDER_COLORS = {"local": "#2563eb", "deepseek": "#f97316", "openai": "#16a34a"}  # blue / orange / green
PROVIDER_LABELS = {
    "local": "Local (Qwen3.5-9B)",
    "deepseek": "DeepSeek API",
    "openai": "OpenAI (ChatGPT)",
}
PROVIDER_DOMAIN = list(PROVIDER_COLORS.keys())
PROVIDER_RANGE = list(PROVIDER_COLORS.values())

st.set_page_config(
    page_title="DE IncidentIQ",
    page_icon="🛠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------ styles ---

BADGE_CSS = """
<style>
.badge {
    display: inline-block;
    padding: 2px 10px;
    margin: 0 6px 4px 0;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1.5;
    white-space: nowrap;
}
.badge-component { background: #1f77b422; color: #1f77b4; border: 1px solid #1f77b455; }
.badge-error     { background: #9467bd22; color: #9467bd; border: 1px solid #9467bd55; }
</style>
"""


def _esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def badges(component: str, error_type: str) -> str:
    out = []
    if component:
        out.append(f'<span class="badge badge-component">{_esc(component)}</span>')
    if error_type:
        out.append(f'<span class="badge badge-error">{_esc(error_type)}</span>')
    return "".join(out)


def _as_list(value) -> list:
    """enriched_documents.json stores `tags` sometimes as a real list, sometimes as its repr."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            return []
    return []


def is_connection_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    needles = (
        "connection",
        "connect",
        "refused",
        "timeout",
        "timed out",
        "max retries",
        "unreachable",
        "apiconnectionerror",
    )
    return any(n in text for n in needles)


def server_error_message(exc: Exception) -> str:
    return (
        f"Cannot reach the LLM/embedding server "
        f"(LLM: {LLM_BASE_URL}, embeddings: {EMBED_BASE_URL}) — "
        f"make sure both llama-server instances are running.\n\nDetails: {exc}"
    )


# ------------------------------------------------------------ cached loads ---


@st.cache_resource(show_spinner="Building indices (keyword + vector)…")
def get_rag(provider: str = "local"):
    """Build the RAG object once per (Streamlit process, provider) pair.
    Cached separately per provider so switching Local <-> DeepSeek <-> OpenAI
    in the sidebar doesn't require restarting the app or rebuilding the
    indices from scratch more than once each."""
    from my_assistant.rag import (
        RAG,
        DEEPSEEK_BASE_URL,
        DEEPSEEK_MODEL,
        LOCAL_LLM_MODEL,
        OPENAI_BASE_URL,
        OPENAI_MODEL,
    )

    if provider == "deepseek":
        return RAG(llm_base_url=DEEPSEEK_BASE_URL, llm_model=DEEPSEEK_MODEL)
    if provider == "openai":
        return RAG(llm_base_url=OPENAI_BASE_URL, llm_model=OPENAI_MODEL)
    # Use this module's own LLM_BASE_URL (reads the LOCAL_LLM_BASE_URL env
    # var, falling back to localhost:11434) rather than rag.py's hardcoded
    # LOCAL_LLM_BASE_URL constant — this is what lets docker-compose point
    # the "local" provider at host.docker.internal from inside a container.
    return RAG(llm_base_url=LLM_BASE_URL, llm_model=LOCAL_LLM_MODEL)


@st.cache_data(show_spinner=False)
def load_doc_count() -> int:
    """Number of documents in the enriched corpus, for the caption."""
    if not DOCS_FILE.exists():
        return 0
    try:
        return len(json.loads(DOCS_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return 0


@st.cache_data(show_spinner=False)
def load_filter_options() -> list[str]:
    """Unique component values from the enriched corpus."""
    if not DOCS_FILE.exists():
        return []
    try:
        docs = json.loads(DOCS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted({(d.get("component") or "").strip() for d in docs} - {""})


# ------------------------------------------------------------- feedback IO ---


def save_feedback(session_id: str, query: str, rating: str, answer_preview: str) -> bool:
    """
    Best-effort append of one feedback row to data/rag_traces.duckdb → rag.feedback (via dlt).
    Never raises: a telemetry failure must not break the UI.
    """
    try:
        import dlt

        TRACE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        @dlt.resource(name="feedback", write_disposition="append")
        def feedback_resource(row: dict):
            yield row

        pipeline = dlt.pipeline(
            pipeline_name="rag_feedback",
            destination=dlt.destinations.duckdb(str(TRACE_DB_PATH)),
            dataset_name="rag",
        )
        pipeline.run(
            feedback_resource(
                {
                    "session_id": session_id,
                    "query": query,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "rating": rating,
                    "answer_preview": (answer_preview or "")[:200],
                }
            )
        )
        return True
    except Exception as exc:
        st.warning(f"Could not save feedback (non-fatal): {exc}")
        return False


# -------------------------------------------------------------- duckdb IO ----


def _connect_readonly():
    import duckdb

    return duckdb.connect(str(TRACE_DB_PATH), read_only=True)


@st.cache_data(ttl=30, show_spinner=False)
def load_monitoring_data() -> dict:
    """
    Read traces + feedback out of the dlt-loaded DuckDB file with plain SQL.

    Returns {"ok": bool, "error": str|None, "traces": DataFrame,
             "feedback": DataFrame, "components": DataFrame}
    Connections are opened and closed immediately so dlt writes are never blocked.
    """
    empty = {
        "ok": False,
        "error": None,
        "traces": pd.DataFrame(),
        "feedback": pd.DataFrame(),
        "components": pd.DataFrame(),
    }

    if not TRACE_DB_PATH.exists():
        empty["error"] = "no_db"
        return empty

    con = None
    try:
        con = _connect_readonly()
        tables = {
            f"{schema}.{name}"
            for schema, name in con.execute(
                "SELECT table_schema, table_name FROM information_schema.tables"
            ).fetchall()
        }

        traces = pd.DataFrame()
        if "rag.rag_traces" in tables:
            traces = con.execute("SELECT * FROM rag.rag_traces").fetchdf()

        feedback = pd.DataFrame()
        if "rag.feedback" in tables:
            feedback = con.execute("SELECT * FROM rag.feedback").fetchdf()

        # dlt unnests list columns into child tables, e.g. rag.rag_traces__top_components
        components = pd.DataFrame()
        child = "rag.rag_traces__top_components"
        if child in tables:
            components = con.execute(
                f"""
                SELECT lower(trim(CAST(value AS VARCHAR))) AS component, COUNT(*) AS mentions
                FROM {child}
                WHERE value IS NOT NULL AND CAST(value AS VARCHAR) <> ''
                GROUP BY 1 ORDER BY mentions DESC
                """
            ).fetchdf()
        elif "top_components" in traces.columns:
            # fallback: the column survived as a native LIST / JSON string
            rows: list[str] = []
            for value in traces["top_components"]:
                if isinstance(value, (list, tuple)):
                    rows.extend(str(v) for v in value)
                else:
                    rows.extend(str(v) for v in _as_list(value))
            rows = [r.strip().lower() for r in rows if str(r).strip()]
            if rows:
                components = (
                    pd.Series(rows, name="component")
                    .value_counts()
                    .rename_axis("component")
                    .reset_index(name="mentions")
                )

        if not traces.empty and "timestamp" in traces.columns:
            traces["timestamp"] = pd.to_datetime(traces["timestamp"], errors="coerce", utc=True)
            traces = traces.dropna(subset=["timestamp"]).sort_values("timestamp")
        if not feedback.empty and "timestamp" in feedback.columns:
            feedback["timestamp"] = pd.to_datetime(feedback["timestamp"], errors="coerce", utc=True)

        return {
            "ok": True,
            "error": None,
            "traces": traces,
            "feedback": feedback,
            "components": components,
        }
    except Exception as exc:  # noqa: BLE001
        empty["error"] = str(exc)
        return empty
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:  # noqa: BLE001
                pass


# ================================================================ PAGE: CHAT ==


def render_chat_page() -> None:
    st.markdown(BADGE_CSS, unsafe_allow_html=True)
    st.title("🛠 DE IncidentIQ")
    doc_count = load_doc_count()
    st.caption(
        "Hybrid retrieval (minsearch keyword + embedding vector search, fused with RRF) "
        f"over {doc_count} curated Spark/data-engineering troubleshooting documents."
    )

    components = load_filter_options()

    with st.sidebar:
        st.subheader("Model")
        provider_options = ["Local (Qwen3.5-9B)", "DeepSeek API", "OpenAI (ChatGPT)"]
        # Default selection follows LLM_PROVIDER from .env if set, else Local.
        default_provider = os.getenv("LLM_PROVIDER", "local").strip().lower()
        default_index = {"local": 0, "deepseek": 1, "openai": 2}.get(default_provider, 0)
        provider_label = st.radio(
            "Answer generation",
            provider_options,
            index=default_index,
            help="Local uses your llama.cpp server (free, needs it running on :11434). "
            "DeepSeek/OpenAI use their cloud APIs (small per-query cost, need "
            "DEEPSEEK_API_KEY / OPENAI_API_KEY set as a real OS/Windows environment "
            "variable — never read from a file). Default choice comes from "
            "LLM_PROVIDER in .env.",
        )
        provider = (
            "deepseek" if provider_label.startswith("DeepSeek")
            else "openai" if provider_label.startswith("OpenAI")
            else "local"
        )
        _provider_key_env = {"deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY"}
        _needed_key = _provider_key_env.get(provider)
        if _needed_key and not os.environ.get(_needed_key):
            st.warning(
                f"{_needed_key} is not set in this environment — {provider_label} calls "
                "will fail. Set it as a real OS/Windows user environment variable, then "
                "restart the app."
            )

        st.subheader("Retrieval")
        retrieval_label = st.radio(
            "Search method",
            [
                "Auto (recommended)",
                "Hybrid — force on",
                "Keyword only — force off",
                "Vector only",
            ],
            index=0,
            help="Auto = use hybrid retrieval if the embedding server (port 11435) responds, "
            "otherwise fall back to keyword-only automatically — no failed queries either way. "
            "Hybrid (force) = keyword + vector fused with RRF, errors if embeddings are down. "
            "Keyword only (force) = pure minsearch/TF-IDF, never touches the embedding model "
            "even if one is running. Vector only = embeddings alone, no keyword leg.",
        )
        retrieval_mode = (
            "auto" if retrieval_label.startswith("Auto")
            else "keyword" if retrieval_label.startswith("Keyword")
            else "vector" if retrieval_label.startswith("Vector")
            else "hybrid"
        )
        if retrieval_mode == "keyword":
            st.caption("✅ No embedding model needed for this query.")
        elif retrieval_mode == "auto":
            st.caption("🔍 Will use embeddings if available, else keyword-only.")

        st.subheader("Filters")
        component = st.selectbox("Component", ["(any)"] + components, index=0)
        n_results = st.slider("Documents to retrieve", 3, 10, 5)
        self_check = st.checkbox(
            "Self-check answer",
            value=True,
            help="Adds a verification pass that reviews the draft answer against the question "
            "and retrieved docs for internal consistency and completeness. If it finds issues, "
            "one repair pass fixes them before you see the answer. Costs 1-2 extra LLM calls.",
        )
        active_llm_url = (
            DEEPSEEK_DISPLAY_URL if provider == "deepseek"
            else OPENAI_DISPLAY_URL if provider == "openai"
            else LLM_BASE_URL
        )
        st.caption(f"LLM: `{active_llm_url}`\n\nEmbeddings: `{EMBED_BASE_URL}`")

    filters: dict = {}
    if component != "(any)":
        filters["component"] = component

    with st.form("query_form"):
        question = st.text_input(
            "Your data engineering question",
            placeholder="e.g. Spark executor OOM during shuffle — how do I fix it?",
        )
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            _run_query(question.strip(), filters or None, n_results, self_check, provider, retrieval_mode)

    if st.session_state.get("last_result"):
        _render_result(st.session_state["last_result"])


def _run_query(
    question: str,
    filters: dict | None,
    n_results: int,
    self_check: bool = True,
    provider: str = "local",
    retrieval_mode: str = "hybrid",
) -> None:
    """Initialise RAG (cached per provider) and run one query, handling server outages gracefully."""
    try:
        rag = get_rag(provider)
    except Exception as exc:  # noqa: BLE001
        get_rag.clear()
        if is_connection_error(exc):
            st.error(server_error_message(exc))
        else:
            st.error(
                "Failed to initialise the RAG pipeline — check that data/enriched_documents.json, "
                f"data/embeddings.npy and data/embeddings_ids.json exist.\n\nDetails: {exc}"
            )
        return

    try:
        spinner_text = (
            "Searching the knowledge base, generating an answer, and self-checking it…"
            if self_check
            else "Searching the knowledge base and generating an answer…"
        )
        with st.spinner(spinner_text):
            result = rag.query(
                question,
                filters=filters,
                n_results=n_results,
                return_docs=True,
                self_check=self_check,
                retrieval_mode=retrieval_mode,
            )
    except Exception as exc:  # noqa: BLE001
        if is_connection_error(exc):
            st.error(server_error_message(exc))
        else:
            st.error(f"Query failed: {exc}")
        return

    # drop the bulky per-doc embedding vectors before parking this in session state
    result["docs"] = [
        {k: v for k, v in doc.items() if k != "embedding"} for doc in result.get("docs", [])
    ]
    st.session_state["last_result"] = result
    st.session_state["feedback_given"] = None
    load_monitoring_data.clear()


def _render_result(result: dict) -> None:
    answer = result.get("answer") or ""
    docs = result.get("docs") or []
    trace = result.get("trace") or {}

    st.markdown("### Answer")
    used_mode = trace.get("retrieval_mode", "hybrid")
    mode_badge = {
        "hybrid": "🔀 Hybrid (keyword + vector)",
        "keyword": "🔤 Keyword only",
        "vector": "🧭 Vector only",
    }.get(used_mode, used_mode)
    st.caption(f"Retrieval used: {mode_badge}")
    st.markdown(answer if answer else "_The model returned an empty answer._")

    if trace.get("out_of_scope"):
        top_score = trace.get("top_vector_score")
        reason = (
            f"best match scored {top_score:.2f}, below the relevance threshold"
            if top_score is not None
            else "no keyword matches at all"
        )
        st.caption(
            f"🚫 No sufficiently relevant documents found ({reason}) — skipped answer "
            "generation entirely rather than risk an ungrounded response."
        )
    elif trace.get("repaired"):
        issues = trace.get("self_check_issues") or []
        with st.expander("🔎 Self-check flagged issues and revised this answer", expanded=False):
            for issue in issues:
                st.markdown(f"- {issue}")
    elif "self_check_flagged" in trace and not trace.get("self_check_flagged"):
        st.caption("✅ Self-check passed — no issues found.")

    # ---- metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tokens", f"{trace.get('total_tokens', 0):,}")
    col2.metric(
        "In / Out",
        f"{trace.get('input_tokens', 0):,} / {trace.get('output_tokens', 0):,}",
    )
    col3.metric("Cost", f"${trace.get('estimated_cost_usd', 0.0):.4f}")
    col4.metric("Latency", f"{trace.get('total_ms', 0.0) / 1000:.1f}s")

    # ---- feedback
    st.markdown("#### Was this helpful?")
    given = st.session_state.get("feedback_given")
    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 8])
    session_id = trace.get("session_id", "")
    up = fb_col1.button("👍 Yes", disabled=given is not None, key="fb_up")
    down = fb_col2.button("👎 No", disabled=given is not None, key="fb_down")

    if up or down:
        rating = "up" if up else "down"
        ok = save_feedback(
            session_id=session_id,
            query=trace.get("query", ""),
            rating=rating,
            answer_preview=trace.get("answer_preview", answer[:200]),
        )
        st.session_state["feedback_given"] = rating
        load_monitoring_data.clear()
        if ok:
            st.toast("Thanks for the feedback!", icon="✅")
        st.rerun()
    elif given:
        fb_col3.success(
            f"Recorded a thumbs-{given} for this answer. Thanks!",
            icon="✅",
        )

    # ---- retrieved documents
    with st.expander(f"Retrieved documents ({len(docs)})", expanded=False):
        if not docs:
            st.info("No documents were retrieved for this query.")
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**{i}. {_esc(doc.get('title') or 'Untitled')}**")
            st.markdown(
                badges(
                    doc.get("component") or "",
                    doc.get("error_type") or "",
                ),
                unsafe_allow_html=True,
            )
            summary = doc.get("summary") or (doc.get("text") or "")[:300]
            if summary:
                st.caption(summary)
            tags = _as_list(doc.get("tags"))
            if tags:
                st.caption("Tags: " + ", ".join(str(t) for t in tags[:10]))
            if doc.get("url"):
                st.caption(f"[Source]({doc['url']})")
            if i < len(docs):
                st.divider()

    with st.expander("Raw trace (debug)", expanded=False):
        st.json(trace)


# ========================================================== PAGE: MONITORING ==


def _empty_state(message: str) -> None:
    st.info(message, icon="📭")
    st.caption(
        "Traces are written automatically by `RAG.query()` via dlt into "
        f"`{TRACE_DB_PATH}` (tables `rag.rag_traces` and `rag.feedback`)."
    )


def render_monitoring_page() -> None:
    st.title("📊 RAG Monitoring Dashboard")
    st.caption(f"Source: `{TRACE_DB_PATH}` (read-only DuckDB)")

    top_l, top_r = st.columns([6, 1])
    with top_r:
        if st.button("🔄 Refresh"):
            load_monitoring_data.clear()
            st.rerun()

    data = load_monitoring_data()

    if data["error"] == "no_db":
        _empty_state("No data yet — run some queries on the Chat page first.")
        return
    if not data["ok"]:
        st.error(f"Could not read the trace database: {data['error']}")
        st.caption(
            "If another process holds a write lock on the DuckDB file, close it and refresh."
        )
        return

    traces: pd.DataFrame = data["traces"]
    feedback: pd.DataFrame = data["feedback"]
    components: pd.DataFrame = data["components"]

    if traces.empty:
        _empty_state("No data yet — run some queries on the Chat page first.")
        return

    _render_summary_cards(traces, feedback)
    st.divider()

    left, right = st.columns(2)
    with left:
        _chart_query_volume(traces)
    with right:
        _chart_latency_breakdown(traces)

    left, right = st.columns(2)
    with left:
        _chart_token_usage(traces)
    with right:
        _chart_cost(traces)

    left, right = st.columns(2)
    with left:
        _chart_feedback(feedback)
    with right:
        _chart_components(components)

    st.divider()
    _render_recent_table(traces)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _render_summary_cards(traces: pd.DataFrame, feedback: pd.DataFrame) -> None:
    total_queries = len(traces)
    avg_latency = _num(traces, "total_ms").mean() / 1000 if total_queries else 0.0
    total_cost = _num(traces, "estimated_cost_usd").sum()
    total_tokens = int(_num(traces, "total_tokens").sum())

    if not feedback.empty and "rating" in feedback.columns:
        ratings = feedback["rating"].astype(str).str.lower()
        ups = int((ratings == "up").sum())
        total_fb = int(ratings.isin(["up", "down"]).sum())
        satisfaction = f"{(ups / total_fb * 100):.0f}%" if total_fb else "n/a"
        fb_help = f"{ups}/{total_fb} thumbs up"
    else:
        satisfaction, fb_help = "n/a", "no feedback yet"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total queries", f"{total_queries:,}")
    c2.metric("Avg latency", f"{avg_latency:.1f}s")
    c3.metric("Total tokens", f"{total_tokens:,}")
    c4.metric("Total cost", f"${total_cost:.4f}")
    c5.metric("Satisfaction", satisfaction, help=fb_help)

    if "llm_provider" in traces.columns:
        by_provider = (
            traces.assign(provider=traces["llm_provider"].fillna("local").astype(str).str.lower())
            .groupby("provider")
            .agg(queries=("provider", "size"), cost=("estimated_cost_usd", lambda s: _num(traces.loc[s.index], "estimated_cost_usd").sum()))
        )
        if len(by_provider) > 1 or "deepseek" in by_provider.index or "openai" in by_provider.index:
            legend_cols = st.columns(len(by_provider))
            for col, (provider, row) in zip(legend_cols, by_provider.iterrows()):
                color = PROVIDER_COLORS.get(provider, "#64748b")
                label = PROVIDER_LABELS.get(provider, provider)
                col.markdown(
                    f"<span style='display:inline-block;width:10px;height:10px;"
                    f"border-radius:50%;background:{color};margin-right:6px;'></span>"
                    f"**{label}** — {int(row['queries'])} queries, ${row['cost']:.4f}",
                    unsafe_allow_html=True,
                )


def _chart_query_volume(traces: pd.DataFrame) -> None:
    st.subheader("1. Query volume over time")
    if "timestamp" not in traces.columns or traces["timestamp"].isna().all():
        st.info("No timestamps available.")
        return
    span_hours = (
        traces["timestamp"].max() - traces["timestamp"].min()
    ).total_seconds() / 3600.0
    freq, label = ("h", "hour") if span_hours <= 72 else ("D", "day")
    st.caption(f"Queries per {label} — colored by provider")

    df = traces.copy()
    df["provider"] = df.get("llm_provider", "local")
    df["provider"] = df["provider"].fillna("local").astype(str).str.lower()

    volume = (
        df.set_index("timestamp")
        .groupby("provider")
        .resample(freq)
        .size()
        .rename("queries")
        .reset_index()
    )

    try:
        import altair as alt

        chart = (
            alt.Chart(volume)
            .mark_bar()
            .encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("queries:Q", title="Queries"),
                color=alt.Color(
                    "provider:N",
                    scale=alt.Scale(domain=PROVIDER_DOMAIN, range=PROVIDER_RANGE),
                    legend=alt.Legend(title="Provider"),
                ),
                tooltip=["provider", "timestamp:T", "queries:Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")
    except Exception:  # noqa: BLE001
        pivot = volume.pivot(index="timestamp", columns="provider", values="queries").fillna(0)
        st.bar_chart(pivot, height=260)


def _chart_latency_breakdown(traces: pd.DataFrame) -> None:
    st.subheader("2. Latency breakdown")
    stages = ["embed_ms", "keyword_ms", "vector_ms", "llm_ms"]
    means = {s.replace("_ms", ""): float(_num(traces, s).mean()) for s in stages}
    df = pd.DataFrame({"stage": list(means), "avg_ms": list(means.values())})
    if df["avg_ms"].sum() == 0:
        st.info("No latency data recorded.")
        return
    try:
        import altair as alt

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("avg_ms:Q", title="Average milliseconds"),
                y=alt.Y("stage:N", sort="-x", title=None),
                color=alt.Color("stage:N", legend=None),
                tooltip=["stage", alt.Tooltip("avg_ms:Q", format=".1f")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")
    except Exception:  # noqa: BLE001
        st.bar_chart(df.set_index("stage"), height=260)
    st.caption(
        "Average time per pipeline stage. Total avg: "
        f"{_num(traces, 'total_ms').mean():.0f} ms"
    )


def _chart_token_usage(traces: pd.DataFrame) -> None:
    st.subheader("3. Token usage per query")
    st.caption("Total tokens per query, in chronological order — colored by provider.")

    df = traces.copy()
    df["total_tokens_num"] = _num(traces, "total_tokens")
    df["provider"] = df.get("llm_provider", "local")
    df["provider"] = df["provider"].fillna("local").astype(str).str.lower()
    if "timestamp" not in df.columns:
        df["timestamp"] = range(len(df))

    if df["total_tokens_num"].sum() == 0:
        st.info("No token usage recorded.")
        return

    try:
        import altair as alt

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T" if "timestamp" in traces.columns else "timestamp:O", title=None),
                y=alt.Y("total_tokens_num:Q", title="Total tokens"),
                color=alt.Color(
                    "provider:N",
                    scale=alt.Scale(domain=PROVIDER_DOMAIN, range=PROVIDER_RANGE),
                    legend=alt.Legend(title="Provider"),
                ),
                tooltip=["provider", "total_tokens_num:Q"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")
    except Exception:  # noqa: BLE001
        pivot = df.pivot_table(
            index="timestamp", columns="provider", values="total_tokens_num", aggfunc="sum"
        ).fillna(0)
        st.line_chart(pivot, height=260)


def _chart_cost(traces: pd.DataFrame) -> None:
    st.subheader("4. Cumulative cost by provider")

    df = traces.copy()
    df["cost_num"] = _num(traces, "estimated_cost_usd")
    df["provider"] = df.get("llm_provider", "local")
    df["provider"] = df["provider"].fillna("local").astype(str).str.lower()
    if "timestamp" not in df.columns:
        df["timestamp"] = range(len(df))
    df = df.sort_values("timestamp")
    df["cumulative_cost_usd"] = df.groupby("provider")["cost_num"].cumsum()

    total = float(df["cost_num"].sum())

    try:
        import altair as alt

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T" if "timestamp" in traces.columns else "timestamp:O", title=None),
                y=alt.Y("cumulative_cost_usd:Q", title="Cumulative cost (USD)"),
                color=alt.Color(
                    "provider:N",
                    scale=alt.Scale(domain=PROVIDER_DOMAIN, range=PROVIDER_RANGE),
                    legend=alt.Legend(title="Provider"),
                ),
                tooltip=["provider", alt.Tooltip("cumulative_cost_usd:Q", format="$.4f")],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")
    except Exception:  # noqa: BLE001
        pivot = df.pivot_table(
            index="timestamp", columns="provider", values="cumulative_cost_usd", aggfunc="max"
        ).ffill().fillna(0)
        st.line_chart(pivot, height=260)

    if total == 0:
        st.caption(
            "Running on a local llama.cpp model — cost is $0.00. "
            "Pricing is applied automatically when a query uses the DeepSeek or OpenAI provider."
        )
    else:
        st.caption(f"Total spend so far: ${total:.4f}")


def _chart_feedback(feedback: pd.DataFrame) -> None:
    st.subheader("5. Feedback ratio")
    if feedback.empty or "rating" not in feedback.columns:
        st.info("No feedback captured yet — use 👍 / 👎 on the Chat page.")
        return
    counts = (
        feedback["rating"].astype(str).str.lower().value_counts().rename_axis("rating")
        .reset_index(name="count")
    )
    counts = counts[counts["rating"].isin(["up", "down"])]
    if counts.empty:
        st.info("No feedback captured yet — use 👍 / 👎 on the Chat page.")
        return
    counts["label"] = counts["rating"].map({"up": "👍 Helpful", "down": "👎 Not helpful"})
    try:
        import altair as alt

        chart = (
            alt.Chart(counts)
            .mark_arc(innerRadius=55)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color(
                    "label:N",
                    title="Rating",
                    scale=alt.Scale(
                        domain=["👍 Helpful", "👎 Not helpful"],
                        range=["#2ca02c", "#d62728"],
                    ),
                ),
                tooltip=["label", "count"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart, width="stretch")
    except Exception:  # noqa: BLE001
        st.bar_chart(counts.set_index("label")["count"], height=260)


def _chart_components(components: pd.DataFrame) -> None:
    st.subheader("6. Components retrieved most often")
    if components.empty:
        st.info("No component data yet.")
        return
    top = components.head(15).set_index("component")["mentions"]
    st.bar_chart(top, height=260)
    st.caption("How often each component appears in the top-5 retrieved docs.")


def _render_recent_table(traces: pd.DataFrame) -> None:
    st.subheader("Recent queries")
    df = traces.copy()
    if "llm_provider" in df.columns:
        provider = df["llm_provider"].fillna("local").astype(str).str.lower()
        emoji = {"local": "🔵", "deepseek": "🟠", "openai": "🟢"}
        df["provider"] = provider.map(lambda p: f"{emoji.get(p, '⚪')} {PROVIDER_LABELS.get(p, p)}")

    cols = [
        c
        for c in [
            "timestamp",
            "provider",
            "query",
            "n_final_docs",
            "total_tokens",
            "estimated_cost_usd",
            "total_ms",
            "llm_model",
            "answer_preview",
        ]
        if c in df.columns
    ]
    recent = df.sort_values("timestamp", ascending=False).head(25)[cols]

    if "provider" in recent.columns:
        st.dataframe(
            recent,
            width="stretch",
            hide_index=True,
            column_config={
                "provider": st.column_config.TextColumn(
                    "Provider",
                    help="🔵 Local (Qwen3.5-9B) — 🟠 DeepSeek API — 🟢 OpenAI (ChatGPT)",
                )
            },
        )
        st.caption("🔵 Local (Qwen3.5-9B) · 🟠 DeepSeek API · 🟢 OpenAI (ChatGPT)")
    else:
        st.dataframe(recent, width="stretch", hide_index=True)


# ======================================================================= main ==


def main() -> None:
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Page",
            ["💬 Chat / Search", "📊 Monitoring"],
            label_visibility="collapsed",
        )
        st.divider()

    if page.endswith("Monitoring"):
        render_monitoring_page()
    else:
        render_chat_page()


main()
