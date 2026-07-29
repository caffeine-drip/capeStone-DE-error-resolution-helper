"""
kb_ingest.py — dlt-based ingestion pipeline: loads the enriched document
corpus into a DuckDB-backed knowledge base table.

This is deliberately a separate, narrower step from enrich_documents.py.
enrich_documents.py does the expensive work (LLM tagging + embedding
generation, ~minutes per run, resumable). kb_ingest.py's job is just to
load the *already enriched* JSON into a real queryable knowledge-base
store via dlt — the "automated ingestion with a special tool" step,
distinct from ad-hoc json.load() calls scattered through the app.

data/enriched_documents.json  -->  dlt  -->  data/knowledge_base.duckdb (kb.documents)

ingest.py's build_indices() reads documents from this DuckDB table when it
exists (auto-running this pipeline on first use if it doesn't), so the
knowledge base genuinely lives in a dlt-managed store, not just a JSON file
re-parsed on every process start. The 1024-dim embedding vectors stay in
data/embeddings.npy (a numpy matrix is the right tool for vector math);
this table holds everything else — text, tags, component, etc.

Usage:
    uv run python my_assistant/kb_ingest.py              # run/refresh the pipeline
    uv run python my_assistant/kb_ingest.py --force      # reload even if content unchanged
    uv run python my_assistant/kb_ingest.py --check      # just report whether a reload is needed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import dlt

ENRICHED_FILE = Path("data/enriched_documents.json")
KB_DB_PATH = Path(os.getenv("KB_DB_PATH", "data/knowledge_base.duckdb"))
STATE_FILE = Path("data/.kb_ingest_state.json")
DATASET_NAME = "kb"
TABLE_NAME = "documents"


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _flatten_for_sql(doc: dict) -> dict:
    """Prepare one enriched document row for a relational table.

    - drops the 1024-dim embedding vector (lives in embeddings.npy instead —
      a numpy matrix is the right storage for vector math, not a SQL column)
    - flattens the tags list into a comma-joined string (simpler than a
      nested/child table for a field nobody needs to query relationally)
    """
    row = dict(doc)
    row.pop("embedding", None)
    tags = row.get("tags")
    if isinstance(tags, list):
        row["tags"] = ",".join(str(t) for t in tags)
    return row


@dlt.resource(name=TABLE_NAME, write_disposition="replace", primary_key="id")
def documents_resource(source_file: Path = ENRICHED_FILE):
    """dlt resource: yields one row per enriched document."""
    docs = json.loads(source_file.read_text(encoding="utf-8"))
    for doc in docs:
        yield _flatten_for_sql(doc)


def needs_reload(force: bool = False) -> bool:
    if force or not KB_DB_PATH.exists():
        return True
    if not STATE_FILE.exists():
        return True
    try:
        prev = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return prev.get("hash") != _content_hash(ENRICHED_FILE)


def run(force: bool = False) -> int:
    if not ENRICHED_FILE.exists():
        print(f"{ENRICHED_FILE} not found — run enrich_documents.py first")
        return 1

    if not needs_reload(force=force):
        print(f"Knowledge base already up to date -> {KB_DB_PATH} ({TABLE_NAME})")
        return 0

    print(f"Ingesting {ENRICHED_FILE} -> {KB_DB_PATH} (dlt pipeline 'kb_ingestion', "
          f"table {DATASET_NAME}.{TABLE_NAME})...")

    pipeline = dlt.pipeline(
        pipeline_name="kb_ingestion",
        destination=dlt.destinations.duckdb(str(KB_DB_PATH)),
        dataset_name=DATASET_NAME,
    )
    load_info = pipeline.run(documents_resource())
    print(load_info)

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps({"hash": _content_hash(ENRICHED_FILE)}), encoding="utf-8"
    )
    print("Done.")
    return 0


def load_documents_from_kb() -> list[dict] | None:
    """
    Read documents back out of the dlt-loaded DuckDB table, for ingest.py to
    consume. Returns None if the KB hasn't been ingested yet (caller should
    fall back to run() then retry, or fall back to reading the JSON directly).
    """
    if not KB_DB_PATH.exists():
        return None
    import duckdb

    con = duckdb.connect(str(KB_DB_PATH), read_only=True)
    try:
        table = f'"{DATASET_NAME}"."{TABLE_NAME}"'
        try:
            df = con.execute(f"SELECT * FROM {table}").fetchdf()
        except Exception:
            return None
    finally:
        con.close()

    docs = df.to_dict(orient="records")
    for doc in docs:
        tags = doc.get("tags")
        if isinstance(tags, str):
            doc["tags"] = [t for t in tags.split(",") if t]
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dlt ingestion: enriched_documents.json -> DuckDB knowledge base")
    parser.add_argument("--force", action="store_true", help="reload even if content hash is unchanged")
    parser.add_argument("--check", action="store_true", help="only report whether a reload is needed")
    args = parser.parse_args()

    if args.check:
        print("needs reload" if needs_reload() else "up to date")
        sys.exit(0)

    sys.exit(run(force=args.force))
