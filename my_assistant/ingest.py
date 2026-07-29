"""
ingest.py — Build keyword and vector indices for the knowledge base.

Documents are loaded from the dlt-managed DuckDB knowledge base
(data/knowledge_base.duckdb, populated by kb_ingest.py) whenever it's
available and up to date. If it isn't — first run, or enriched_documents.json
changed since the last ingest — this module runs kb_ingest.py itself so
`RAG()` still just works, then reads from the freshly loaded table. Falls
back to reading enriched_documents.json directly only if dlt/DuckDB are
unavailable for some reason.

Usage:
    from my_assistant.ingest import build_indices
    keyword_index, vector_store = build_indices()
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from minsearch import Index

# Lives inside the my_assistant/ package but repo root needs to be on
# sys.path for `from my_assistant import kb_ingest` to resolve when this
# file is run directly (e.g. `uv run python my_assistant/ingest.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ENRICHED_FILE = Path("data/enriched_documents.json")
EMBEDDINGS_FILE = Path("data/embeddings.npy")
IDS_FILE = Path("data/embeddings_ids.json")


def load_documents(path: Path = ENRICHED_FILE) -> list[dict]:
    """
    Load documents from the dlt-ingested knowledge base (preferred), running
    the ingestion pipeline first if it's missing or stale. Falls back to
    reading the JSON directly if kb_ingest/dlt/duckdb aren't usable.
    """
    try:
        from my_assistant import kb_ingest

        if kb_ingest.needs_reload():
            kb_ingest.run()
        docs = kb_ingest.load_documents_from_kb()
        if docs:
            return docs
    except Exception as exc:  # noqa: BLE001 — any failure here just falls back
        warnings.warn(
            f"dlt knowledge-base ingestion unavailable ({exc}) — "
            f"reading {path} directly instead.",
            stacklevel=1,
        )

    return json.loads(path.read_text(encoding="utf-8"))


def build_keyword_index(docs: list[dict]) -> Index:
    """minsearch full-text index over the enriched fields."""
    index = Index(
        text_fields=["title", "summary", "text", "tags_str", "component", "error_type"],
        keyword_fields=["source", "component", "error_type"],
    )
    # Flatten tags list → space-joined string for text search
    prepared = []
    for doc in docs:
        d = dict(doc)
        d["tags_str"] = " ".join(doc.get("tags") or [])
        d.pop("embedding", None)          # keep index lean
        prepared.append(d)
    index.fit(prepared)
    return index


class VectorStore:
    """Cosine-similarity search over the pre-computed embedding matrix."""

    def __init__(self, matrix: np.ndarray, ids: list[str], docs: list[dict]):
        # L2-normalise once so cosine sim = dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = (matrix / norms).astype(np.float32)
        self.ids = ids
        self._id_to_doc = {doc["id"]: doc for doc in docs}

    def search(self, query_vector: list[float], n: int = 10) -> list[dict]:
        q = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores = self.matrix @ q
        top_idx = np.argsort(scores)[::-1][:n]
        results = []
        for i in top_idx:
            doc_id = self.ids[i]
            doc = self._id_to_doc.get(doc_id)
            if doc:
                results.append({**doc, "_score": float(scores[i])})
        return results


def build_indices(path: Path = ENRICHED_FILE) -> tuple[Index, VectorStore]:
    docs = load_documents(path)
    print(f"Loaded {len(docs)} documents")

    keyword_index = build_keyword_index(docs)
    print("Keyword index built")

    matrix = np.load(EMBEDDINGS_FILE).astype(np.float32)
    ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    vector_store = VectorStore(matrix, ids, docs)
    print(f"Vector store built: {matrix.shape}")

    return keyword_index, vector_store


if __name__ == "__main__":
    build_indices()
    print("Done.")
