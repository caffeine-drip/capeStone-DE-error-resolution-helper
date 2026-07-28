"""
enrich_documents.py — LLM metadata enrichment + embedding generation.

For every document in data/all_documents.json this script:
  1. asks the local Qwen model for structured metadata
     (summary / component / error_type / severity / tags),
  2. generates a 1024-dim embedding from the local embeddings server,
and writes:
  data/enriched_documents.json  — documents with metadata + `embedding`
  data/embeddings.npy           — float32 (n_docs, 1024) matrix
  data/embeddings_ids.json      — row -> doc id mapping (keeps .npy aligned)

Runs concurrently, retries on failure, and is resumable — rerun after a crash
and it picks up where it left off.

Usage:
    python enrich_documents.py
    python enrich_documents.py --workers 8 --limit 50
    python enrich_documents.py --rebuild-npy      # only re-export the .npy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------- config ----

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11435/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "local")
EMBED_MODEL = os.getenv("EMBED_MODEL", "local")
EMBED_DIM = 1024

INPUT_FILE = Path("data/all_documents.json")
OUTPUT_FILE = Path("data/enriched_documents.json")
NPY_FILE = Path("data/embeddings.npy")
IDS_FILE = Path("data/embeddings_ids.json")

DEFAULT_WORKERS = 1  # sequential — single slot, max context per call
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.5      # seconds; exponential backoff
CHECKPOINT_EVERY = 25
REQUEST_TIMEOUT = 180  # LLM with thinking can be slow; Ctrl+C stops between retries
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))  # metadata JSON is ~150 tokens
# When thinking cannot be turned off, the JSON only appears after the think
# block, so the budget has to cover the reasoning too.
MAX_TOKENS_THINKING = int(os.getenv("LLM_MAX_TOKENS_THINKING", "4000"))
DEBUG = os.getenv("ENRICH_DEBUG", "").lower() in {"1", "true", "yes"}

VALID_COMPONENTS = {
    "spark", "kafka", "airflow", "redshift", "aws-emr", "schema", "data-quality",
    "security", "cloud-cost", "backfill", "streaming", "orchestration", "general",
}
VALID_ERROR_TYPES = {
    "oom", "performance", "data-loss", "connectivity", "configuration",
    "schema-mismatch", "data-quality", "security", "cost", "dependency-failure", "general",
}
VALID_SEVERITIES = {"high", "medium", "low"}

SYSTEM_PROMPT = """You are a data engineering expert. Given a document about a DE/Spark/data issue,
extract structured metadata as JSON. Return ONLY valid JSON, no explanation, no markdown fences.

Required fields:
- summary: one clear sentence describing the problem and fix
- component: primary system involved (choose one: spark, kafka, airflow, redshift, aws-emr,
             schema, data-quality, security, cloud-cost, backfill, streaming, orchestration, general)
- error_type: category of error (choose one: oom, performance, data-loss, connectivity,
              configuration, schema-mismatch, data-quality, security, cost, dependency-failure, general)
- severity: high / medium / low
- tags: list of 3-8 lowercase technical keywords (e.g. ["shuffle", "data-skew", "executor", "oom"])
"""

# --------------------------------------------------------------- clients ----

llm_client = OpenAI(api_key="local", base_url=LLM_BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)
embed_client = OpenAI(api_key="local", base_url=EMBED_BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)

_print_lock = threading.Lock()
_state_lock = threading.Lock()
_stop_event = threading.Event()


def warn(msg: str) -> None:
    with _print_lock:
        tqdm.write(msg)


# ------------------------------------------------------- json extraction ----

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(raw: str) -> dict | None:
    """Pull a JSON object out of a model response.

    Handles: bare JSON, ```json fenced blocks, leftover <think> blocks,
    leading prose, trailing commas, and single-quoted keys.
    """
    if not raw:
        return None

    text = _THINK_RE.sub("", raw).strip()
    # Unterminated <think> (hit max_tokens) — drop everything before the close.
    if "<think>" in text:
        text = text.split("</think>")[-1].strip() if "</think>" in text else ""

    candidates: list[str] = []

    fenced = _FENCE_RE.findall(text)
    candidates.extend(block.strip() for block in fenced)
    candidates.append(text)

    # Balanced-brace scan: the outermost {...} in the response.
    start = text.find("{")
    if start != -1:
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
                        candidates.append(text[start : i + 1])
                        break

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate.startswith("{"):
            continue
        for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


# ------------------------------------------------------------ validation ----


def clean_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[,;]", value)
    if not isinstance(value, (list, tuple, set)):
        return []
    tags: list[str] = []
    for tag in value:
        tag = re.sub(r"[^a-z0-9\-_. ]", "", str(tag).strip().lower()).strip().replace(" ", "-")
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:8]


def coerce(value: Any, allowed: set[str], default: str) -> str:
    candidate = str(value or "").strip().lower().replace("_", "-")
    if candidate in allowed:
        return candidate
    # tolerate near-misses like "AWS EMR", "out-of-memory", "Spark (EMR)"
    for option in allowed:
        if option != "general" and option in candidate:
            return option
    return default


def normalize_metadata(meta: dict, doc: dict) -> dict:
    summary = str(meta.get("summary") or "").strip()
    if not summary:
        summary = doc.get("title", "").strip()
    return {
        "summary": summary[:500],
        "component": coerce(meta.get("component"), VALID_COMPONENTS, "general"),
        "error_type": coerce(meta.get("error_type"), VALID_ERROR_TYPES, "general"),
        "severity": coerce(meta.get("severity"), VALID_SEVERITIES, "medium"),
        "tags": clean_tags(meta.get("tags")),
    }


def fallback_metadata(doc: dict) -> dict:
    return {
        "summary": doc.get("title", ""),
        "component": "general",
        "error_type": "general",
        "severity": "medium",
        "tags": [],
    }


# ----------------------------------------------------------- LLM / embed ----


def build_user_message(doc: dict) -> str:
    parts = [f"Title: {doc.get('title', '')}"]
    if doc.get("problem"):
        parts.append(f"Problem: {doc['problem'][:3000]}")
    elif doc.get("text"):
        parts.append(f"Content: {doc['text'][:3000]}")
    if doc.get("root_cause"):
        parts.append(f"Root Cause: {doc['root_cause'][:2000]}")
    if doc.get("resolution"):
        parts.append(f"Resolution: {doc['resolution'][:2000]}")
    if doc.get("key_learnings"):
        parts.append(f"Key Learnings: {doc['key_learnings'][:1000]}")
    return "\n\n".join(parts)


def with_retries(fn, label: str, doc_id: str):
    """Run fn() with exponential backoff. Returns None if all attempts fail."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        if _stop_event.is_set():
            return None
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — local server can fail many ways
            last_error = exc
            if attempt < MAX_RETRIES and not _stop_event.is_set():
                time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    warn(f"  {label} FAILED after {MAX_RETRIES} attempts [{doc_id}]: {last_error}")
    warn(f"    └─ context snippet: {str(last_error)[:300]}")
    return None


# -- thinking-mode negotiation -------------------------------------------------
#
# Qwen3.x GGUFs ship with thinking enabled in the chat template. llama-server
# then ends the prompt with an *open* `<think>` tag, and its reasoning parser
# routes every generated token into `reasoning_content` until it sees `</think>`.
# If the model never closes the block within the token budget, `content` comes
# back as an empty string forever — which is the bug we hit.
#
# There is no single flag that works across llama.cpp builds, so we try a ladder
# of request shapes once, keep the first one that yields parseable JSON, and use
# it for the remaining documents.
#
# `chat_template_kwargs` / `reasoning_format` require the server to run with
# `--jinja`. Without it those requests fail (or are ignored) and we fall through
# to the `/no_think` soft switch or to reading `reasoning_content` directly.

METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "component": {"type": "string", "enum": sorted(VALID_COMPONENTS)},
        "error_type": {"type": "string", "enum": sorted(VALID_ERROR_TYPES)},
        "severity": {"type": "string", "enum": sorted(VALID_SEVERITIES)},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
    },
    "required": ["summary", "component", "error_type", "severity", "tags"],
}

# (name, extra request kwargs, suffix appended to the user message)
STRATEGIES: list[tuple[str, dict, str]] = [
    # 1. Ask the template to disable thinking. Qwen uses `enable_thinking`;
    #    some templates use `thinking`. Passing both is harmless.
    ("no-think-kwargs", {"extra_body": {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}}}, ""),
    # 2. Same, plus tell llama.cpp not to split reasoning out of content at all.
    ("no-think+reasoning-none", {"extra_body": {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}, "reasoning_format": "none"}}, ""),
    # 3. Keep thinking off the response by forcing raw content only.
    ("reasoning-none", {"extra_body": {"reasoning_format": "none"}}, ""),
    # 4. Qwen soft switch — works even without --jinja.
    ("no-think-soft-switch", {}, "\n\n/no_think"),
    # 5. Constrain the output to the schema; the grammar makes the model emit
    #    JSON immediately instead of a think block.
    ("json-schema", {"extra_body": {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}},
                     "response_format": {"type": "json_schema",
                                         "json_schema": {"name": "doc_metadata", "strict": True,
                                                         "schema": METADATA_SCHEMA}}}, ""),
    # 6. Plain request — relies on reading reasoning_content / stripping <think>.
    ("plain", {}, ""),
]

# Strategies that leave thinking on need room for the think block itself.
THINKING_STRATEGIES = {"no-think-soft-switch", "plain"}

_strategy_lock = threading.Lock()
_active_strategy: int | None = None


def budget_for(name: str) -> int:
    return MAX_TOKENS_THINKING if name in THINKING_STRATEGIES else MAX_TOKENS


def message_text(msg: Any) -> tuple[str, str]:
    """Return (content, reasoning) from a chat message, tolerating extra fields."""
    content = (getattr(msg, "content", None) or "").strip()
    reasoning = ""
    for field in ("reasoning_content", "reasoning", "thinking"):
        value = getattr(msg, field, None)
        if isinstance(value, str) and value.strip():
            reasoning = value.strip()
            break
    if not reasoning:
        try:  # openai>=1 pydantic models keep unknown keys in model_extra
            extra = msg.model_extra or {}
        except Exception:  # noqa: BLE001
            extra = {}
        for field in ("reasoning_content", "reasoning", "thinking"):
            value = extra.get(field)
            if isinstance(value, str) and value.strip():
                reasoning = value.strip()
                break
    return content, reasoning


def call_llm(user_message: str, strategy_index: int) -> tuple[dict | None, str]:
    """One chat completion using STRATEGIES[strategy_index].

    Returns (parsed_json_or_None, debug_note). Raises on transport errors.
    """
    name, kwargs, suffix = STRATEGIES[strategy_index]
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message + suffix},
        ],
        temperature=0.1,
        max_tokens=budget_for(name),
        **kwargs,
    )
    choice = response.choices[0]
    content, reasoning = message_text(choice.message)

    # Prefer real content; fall back to the reasoning channel, which is where
    # llama.cpp parks everything when a think block is left open.
    parsed = extract_json(content) if content else None
    if parsed is None and reasoning:
        parsed = extract_json(reasoning)
    note = (f"strategy={name} finish={choice.finish_reason} "
            f"content_len={len(content)} reasoning_len={len(reasoning)}")
    if parsed is None:
        note += f" preview={(content or reasoning)[:200]!r}"
    return parsed, note


def pick_strategy(user_message: str, doc_id: str) -> tuple[dict | None, int | None]:
    """Walk the ladder until one request shape returns parseable JSON."""
    notes: list[str] = []
    for index, (name, _, _) in enumerate(STRATEGIES):
        try:
            parsed, note = call_llm(user_message, index)
        except Exception as exc:  # noqa: BLE001 — unsupported flag, 400, timeout...
            notes.append(f"{name}: request failed -> {str(exc)[:160]}")
            continue
        notes.append(note)
        if parsed is not None:
            warn(f"  [llm] using request strategy '{name}' for the rest of the run")
            return parsed, index
    warn(f"  [llm] no working strategy for doc {doc_id}; tried:")
    for note in notes:
        warn(f"        - {note}")
    warn("        Hint: restart llama-server with --jinja (and optionally "
         "--reasoning-format none) so chat_template_kwargs are honoured.")
    return None, None


def enrich_doc(doc: dict) -> dict:
    global _active_strategy
    user_message = build_user_message(doc)

    def call() -> dict:
        global _active_strategy
        with _strategy_lock:
            index = _active_strategy

        if index is None:
            parsed, chosen = pick_strategy(user_message, str(doc.get("id")))
            if chosen is not None:
                with _strategy_lock:
                    if _active_strategy is None:
                        _active_strategy = chosen
            if parsed is None:
                raise ValueError("no request strategy produced parseable JSON")
            return parsed

        parsed, note = call_llm(user_message, index)
        if parsed is None:
            # The chosen shape stopped working (e.g. this doc blew the budget).
            warn(f"  [empty/unparseable] id={doc.get('id')} {note}")
            parsed, chosen = pick_strategy(user_message, str(doc.get("id")))
            if chosen is not None:
                with _strategy_lock:
                    _active_strategy = chosen
            if parsed is None:
                raise ValueError(f"unparseable response ({note})")
        elif DEBUG:
            warn(f"  [ok] id={doc.get('id')} {note}")
        return parsed

    meta = with_retries(call, "enrich", str(doc.get("id")))
    return normalize_metadata(meta, doc) if meta else fallback_metadata(doc)


def diagnose() -> int:
    """Live probe: show exactly what the server returns for each request shape."""
    import urllib.request

    print(f"LLM server: {LLM_BASE_URL}")
    props_url = LLM_BASE_URL.rstrip("/").removesuffix("/v1") + "/props"
    try:
        with urllib.request.urlopen(props_url, timeout=10) as handle:
            props = json.loads(handle.read().decode("utf-8"))
        template = str(props.get("chat_template", ""))[:200]
        print(f"  model path : {props.get('model_path')}")
        print(f"  n_ctx      : {props.get('default_generation_settings', {}).get('n_ctx')}")
        print(f"  chat template head: {template!r}")
        print(f"  jinja/template detected: {'yes' if template else 'no'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not read {props_url}: {exc})")

    probe = ("Title: Spark job fails with OutOfMemoryError during shuffle\n\n"
             "Problem: Executors die with java.lang.OutOfMemoryError on a wide join.\n\n"
             "Resolution: Increased spark.sql.shuffle.partitions and enabled AQE skew join.")

    working: list[str] = []
    for index, (name, kwargs, suffix) in enumerate(STRATEGIES):
        print(f"\n--- strategy {index}: {name} ---")
        print(f"    extra: {json.dumps(kwargs)[:200]}  suffix={suffix!r}")
        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": probe + suffix},
                ],
                temperature=0.1,
                max_tokens=budget_for(name),
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    REQUEST FAILED: {str(exc)[:400]}")
            continue

        print("    raw response:")
        print(json.dumps(response.model_dump(), indent=2, default=str)[:2500])
        choice = response.choices[0]
        content, reasoning = message_text(choice.message)
        parsed = extract_json(content) or (extract_json(reasoning) if reasoning else None)
        print(f"    finish_reason={choice.finish_reason} content_len={len(content)} "
              f"reasoning_len={len(reasoning)} parsed={'YES' if parsed else 'NO'}")
        if parsed:
            print(f"    parsed JSON: {json.dumps(parsed)[:400]}")
            working.append(name)

    print("\n==============================")
    if working:
        print(f"Working strategies: {', '.join(working)}")
        print("enrich_documents.py will auto-select the first one at runtime.")
    else:
        print("NOTHING worked. Restart llama-server with --jinja, e.g.:")
        print("  llama-server.exe -m Qwen3.5-9B-Q4_K_M.gguf --port 11434 --ctx-size 32768 "
              "-ngl 999 --flash-attn on --parallel 1 --batch-size 2048 --jinja "
              "--reasoning-format none")
        print("--jinja is required for chat_template_kwargs / response_format to be honoured.")
    return 0 if working else 1


def embed_text(doc: dict) -> list[float] | None:
    text = doc.get("text") or doc.get("title") or ""
    # Prepend the LLM summary — it sharpens retrieval on short/noisy docs.
    summary = doc.get("summary") or ""
    payload = f"{doc.get('title', '')}\n{summary}\n{text}".strip()[:6000]
    if not payload:
        return None

    def call() -> list[float]:
        response = embed_client.embeddings.create(model=EMBED_MODEL, input=payload)
        vector = response.data[0].embedding
        if len(vector) != EMBED_DIM:
            raise ValueError(f"expected {EMBED_DIM} dims, got {len(vector)}")
        return [float(v) for v in vector]

    return with_retries(call, "embed", str(doc.get("id")))


def process_doc(doc: dict) -> dict:
    enriched = dict(doc)
    enriched.update(enrich_doc(doc))
    embedding = embed_text(enriched)
    enriched["embedding"] = embedding or []
    return enriched


# ----------------------------------------------------------------- I/O -----


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def export_embeddings(docs: list[dict]) -> None:
    rows, ids = [], []
    for doc in docs:
        vector = doc.get("embedding")
        if isinstance(vector, list) and len(vector) == EMBED_DIM:
            rows.append(vector)
            ids.append(doc["id"])

    if not rows:
        print("No embeddings to export — is the embeddings server on 11435 running?")
        return

    matrix = np.asarray(rows, dtype=np.float32)
    NPY_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(NPY_FILE, matrix)
    atomic_write_json(IDS_FILE, ids)
    print(f"Saved embeddings matrix {matrix.shape} -> {NPY_FILE}")
    print(f"Saved row->id mapping ({len(ids)} ids) -> {IDS_FILE}")


def health_check() -> None:
    for name, client, base in (("LLM", llm_client, LLM_BASE_URL), ("embeddings", embed_client, EMBED_BASE_URL)):
        try:
            client.models.list()
            print(f"  {name} server OK  ({base})")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: {name} server unreachable at {base}: {exc}")


# ---------------------------------------------------------------- main -----


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich documents with LLM metadata and embeddings.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="concurrent requests (4-8)")
    parser.add_argument("--limit", type=int, default=None, help="process at most N docs (smoke test)")
    parser.add_argument("--input", default=str(INPUT_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--force", action="store_true", help="ignore existing output and re-enrich everything")
    parser.add_argument("--rebuild-npy", action="store_true", help="only re-export .npy from existing output")
    parser.add_argument("--diagnose", action="store_true",
                        help="probe the LLM server with every request shape and print raw responses")
    args = parser.parse_args()

    if args.diagnose:
        return diagnose()

    input_file, output_file = Path(args.input), Path(args.output)

    if args.rebuild_npy:
        if not output_file.exists():
            print(f"{output_file} not found")
            return 1
        export_embeddings(json.loads(output_file.read_text(encoding="utf-8")))
        return 0

    if not input_file.exists():
        print(f"{input_file} not found — run merge_data.py first")
        return 1

    docs = json.loads(input_file.read_text(encoding="utf-8"))
    print(f"Loaded {len(docs)} documents from {input_file}")
    health_check()

    enriched: list[dict] = []
    if output_file.exists() and not args.force:
        try:
            enriched = json.loads(output_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("Existing output is corrupt — starting fresh")
            enriched = []
        # Only treat a doc as done if it has BOTH metadata and an embedding.
        done_ids = {
            d["id"]
            for d in enriched
            if d.get("summary") and len(d.get("embedding") or []) == EMBED_DIM
        }
        enriched = [d for d in enriched if d["id"] in done_ids]
        pending = [d for d in docs if d["id"] not in done_ids]
        print(f"Resuming — {len(enriched)} complete, {len(pending)} remaining")
    else:
        pending = docs

    if args.limit:
        pending = pending[: args.limit]

    workers = max(1, min(args.workers, 16))
    start = time.time()
    processed_since_checkpoint = 0

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_doc, doc): doc for doc in pending}
            bar = tqdm(as_completed(futures), total=len(futures), desc="Enriching", unit="doc")
            try:
                for future in bar:
                    doc = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        warn(f"  UNEXPECTED failure [{doc.get('id')}]: {exc}")
                        result = {**doc, **fallback_metadata(doc), "embedding": []}

                    with _state_lock:
                        enriched.append(result)
                        processed_since_checkpoint += 1
                        if processed_since_checkpoint >= CHECKPOINT_EVERY:
                            atomic_write_json(output_file, enriched)
                            processed_since_checkpoint = 0
                            bar.set_postfix_str(f"checkpoint @ {len(enriched)}")
            except KeyboardInterrupt:
                print("\nInterrupted — stopping...")
                _stop_event.set()
                pool.shutdown(wait=False, cancel_futures=True)
                with _state_lock:
                    atomic_write_json(output_file, enriched)
                    export_embeddings(enriched)
                return 130

    atomic_write_json(output_file, enriched)
    export_embeddings(enriched)

    # ------------------------------------------------------- summary ----
    elapsed = time.time() - start
    missing_embed = sum(1 for d in enriched if len(d.get("embedding") or []) != EMBED_DIM)
    fallbacks = sum(1 for d in enriched if d.get("component") == "general" and not d.get("tags"))

    print(f"\nDone in {elapsed:.1f}s — {len(enriched)} enriched documents -> {output_file}")
    if pending:
        print(f"Throughput: {len(pending) / max(elapsed, 0.001):.2f} docs/s with {workers} workers")
    print(f"Docs missing embeddings: {missing_embed}")
    print(f"Docs that fell back to default metadata: {fallbacks}")

    for field in ("component", "error_type", "severity"):
        counts: dict[str, int] = {}
        for doc in enriched:
            counts[doc.get(field, "?")] = counts.get(doc.get(field, "?"), 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        print(f"{field}: " + ", ".join(f"{k}={v}" for k, v in top[:12]))

    return 0


if __name__ == "__main__":
    sys.exit(main())