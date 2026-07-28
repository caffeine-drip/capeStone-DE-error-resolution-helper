"""
ingest.py — Build keyword and vector indices from enriched_documents.json.

Usage:
    from ingest import build_indices
    keyword_index, vector_store = build_indices()
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from minsearch import Index

ENRICHED_FILE = Path("data/enriched_documents.json")
EMBEDDINGS_FILE = Path("data/embeddings.npy")
IDS_FILE = Path("data/embeddings_ids.json")


def load_documents(path: Path = ENRICHED_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_keyword_index(docs: list[dict]) -> Index:
    """minsearch full-text index over the enriched fields."""
    index = Index(
        text_fields=["title", "summary", "text", "tags_str", "component", "error_type"],
        keyword_fields=["source", "component", "error_type", "severity"],
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
