"""
merge_data.py — merge, normalize and quality-filter all capstone source documents.

Reads the raw per-source JSON files, strips HTML (Stack Overflow bodies),
guarantees every document has a flat `text` field, drops junk/duplicate docs,
and writes a single normalized corpus to data/all_documents.json.

Usage:
    python merge_data.py
    python merge_data.py --min-chars 80 --out data/all_documents.json
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------- config ----

SOURCE_FILES: list[tuple[str, str]] = [
    # (path, fallback source label if the doc has no `source` field)
    ("data/documents.json", "databricks-kb"),
    ("data/pdf_scenarios.json", "pdf-scenario"),
    ("data/emr_spark.json", "aws-emr"),
    ("data/so_spark.json", "stackoverflow"),
]

DEFAULT_OUTPUT = "data/all_documents.json"
MIN_TEXT_CHARS = 50

# Fields that get concatenated into `text` when a doc has no usable flat text.
NARRATIVE_FIELDS = ["problem", "root_cause", "resolution", "key_learnings"]

FIELD_LABELS = {
    "problem": "Problem",
    "root_cause": "Root Cause",
    "resolution": "Resolution",
    "key_learnings": "Key Learnings",
}

# ------------------------------------------------------------ html strip ----

try:
    from bs4 import BeautifulSoup

    _HAS_BS4 = True
except ImportError:  # graceful degradation — regex fallback
    _HAS_BS4 = False

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_HTML_HINT_RE = re.compile(r"<(p|div|a|code|pre|br|ul|ol|li|blockquote|h[1-6]|strong|em)\b", re.I)


def looks_like_html(text: str) -> bool:
    return bool(_HTML_HINT_RE.search(text))


def strip_html(text: str) -> str:
    """Convert an HTML fragment (typical Stack Overflow body) to clean plain text.

    Code blocks are preserved as fenced blocks so the LLM/embedder still sees
    stack traces and configuration snippets, which carry most of the signal.
    """
    if not text:
        return ""
    if not looks_like_html(text):
        return html.unescape(text)

    if not _HAS_BS4:
        return html.unescape(_TAG_RE.sub(" ", text))

    soup = BeautifulSoup(text, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    # Fence <pre> blocks (SO wraps code in <pre><code>).
    for pre in soup.find_all("pre"):
        code = pre.get_text("\n", strip=False).strip("\n")
        pre.replace_with(f"\n```\n{code}\n```\n")

    # Inline code -> backticks.
    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text(strip=True)}`")

    for br in soup.find_all("br"):
        br.replace_with("\n")

    for block in soup.find_all(["p", "li", "div", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]):
        block.append("\n")

    return soup.get_text()


def normalize_text(text: Any) -> str:
    """Strip HTML, unescape entities, collapse whitespace — order matters."""
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = "\n".join(f"- {normalize_text(item)}" for item in text if item)
    if not isinstance(text, str):
        text = str(text)

    text = strip_html(text)
    text = html.unescape(text)
    text = text.replace(" ", " ").replace("​", "")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


# -------------------------------------------------------------- building ----


def build_text(doc: dict) -> str:
    """Guarantee a flat, self-contained `text` field.

    PDF scenario docs have problem/root_cause/resolution/key_learnings but no
    flat text; KB and SO docs have text but benefit from the title prefix.
    """
    title = normalize_text(doc.get("title"))
    flat = normalize_text(doc.get("text"))

    sections: list[str] = []
    for field in NARRATIVE_FIELDS:
        value = normalize_text(doc.get(field))
        if value:
            sections.append(f"{FIELD_LABELS[field]}: {value}")

    parts: list[str] = []
    if title:
        parts.append(title)

    if sections:
        # Structured doc — prefer the structured sections, append flat text only
        # if it adds something the sections don't already contain.
        parts.extend(sections)
        if flat and flat[:200] not in "\n".join(sections):
            parts.append(flat)
    elif flat:
        parts.append(flat)

    return "\n\n".join(p for p in parts if p).strip()


def content_fingerprint(text: str) -> str:
    """Hash of aggressively normalized text, for near-duplicate detection."""
    squashed = re.sub(r"[^a-z0-9]+", "", text.lower())
    return hashlib.sha1(squashed.encode("utf-8")).hexdigest()


def make_id(path: Path, index: int, doc: dict) -> str:
    """Deterministic fallback id for docs missing one."""
    basis = f"{path.stem}:{index}:{doc.get('title', '')}"
    return f"{path.stem}-{hashlib.sha1(basis.encode('utf-8')).hexdigest()[:12]}"


def load_json_docs(path: Path) -> list[dict]:
    """Accept either a JSON array or JSONL, and tolerate a {"documents": [...]} wrapper."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    if isinstance(data, dict):
        for key in ("documents", "docs", "data", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    return [d for d in data if isinstance(d, dict)]


# ------------------------------------------------------------------ main ----


def merge(files: Iterable[tuple[str, str]], out_path: Path, min_chars: int) -> list[dict]:
    all_docs: list[dict] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    stats: dict[str, Counter] = defaultdict(Counter)
    topic_counts: dict[str, Counter] = defaultdict(Counter)
    char_totals: Counter = Counter()

    for file_path, default_source in files:
        path = Path(file_path)
        label = default_source

        if not path.exists():
            print(f"SKIP (not found): {file_path}")
            stats[label]["missing_file"] = 1
            continue

        try:
            docs = load_json_docs(path)
        except Exception as exc:  # noqa: BLE001 — one bad file shouldn't kill the run
            print(f"SKIP (unreadable: {exc}): {file_path}")
            stats[label]["unreadable"] = 1
            continue

        for index, doc in enumerate(docs):
            source = str(doc.get("source") or default_source)
            stats[source]["read"] += 1

            doc_id = str(doc.get("id") or "").strip() or make_id(path, index, doc)

            if doc_id in seen_ids:
                stats[source]["dup_id"] += 1
                continue

            text = build_text(doc)
            if len(text) < min_chars:
                stats[source]["too_short"] += 1
                continue

            fingerprint = content_fingerprint(text)
            if fingerprint in seen_hashes:
                stats[source]["dup_text"] += 1
                continue

            clean = dict(doc)
            clean["id"] = doc_id
            clean["source"] = source
            clean["title"] = normalize_text(doc.get("title")) or text[:80]
            clean["text"] = text
            clean["topic"] = normalize_text(doc.get("topic")) or "general"
            for field in NARRATIVE_FIELDS + ["category"]:
                if field in clean:
                    clean[field] = normalize_text(clean[field])
            clean["char_len"] = len(text)
            clean["word_len"] = len(text.split())

            seen_ids.add(doc_id)
            seen_hashes.add(fingerprint)
            all_docs.append(clean)

            stats[source]["kept"] += 1
            char_totals[source] += len(text)
            topic_counts[source][clean["topic"]] += 1

        kept_here = sum(v["kept"] for v in stats.values())
        print(f"{file_path}: read {len(docs)} -> corpus now {kept_here}")

    # -------------------------------------------------------- report ----
    print("\n" + "=" * 78)
    print(f"{'SOURCE':<22}{'READ':>7}{'KEPT':>7}{'DUP-ID':>8}{'DUP-TXT':>9}{'SHORT':>7}{'AVG CHARS':>11}")
    print("-" * 78)
    for source in sorted(stats):
        row = stats[source]
        kept = row["kept"]
        avg = char_totals[source] // kept if kept else 0
        print(
            f"{source[:22]:<22}{row['read']:>7}{kept:>7}"
            f"{row['dup_id']:>8}{row['dup_text']:>9}{row['too_short']:>7}{avg:>11}"
        )
    print("-" * 78)
    total_read = sum(r["read"] for r in stats.values())
    total_kept = len(all_docs)
    print(f"{'TOTAL':<22}{total_read:>7}{total_kept:>7}")
    print("=" * 78)

    if all_docs:
        lengths = sorted(d["char_len"] for d in all_docs)
        print(
            f"text length — min {lengths[0]} / "
            f"median {lengths[len(lengths) // 2]} / "
            f"p90 {lengths[int(len(lengths) * 0.9)]} / max {lengths[-1]} chars"
        )
        top_topics = Counter(d["topic"] for d in all_docs).most_common(10)
        print("top topics: " + ", ".join(f"{t} ({c})" for t, c in top_topics))

    if not _HAS_BS4:
        print("\nWARNING: beautifulsoup4 not installed — used regex HTML fallback.")
        print("         pip install beautifulsoup4  (recommended for Stack Overflow bodies)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(all_docs, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_path)
    print(f"\nSaved {total_kept} documents to {out_path}")
    return all_docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and normalize capstone source documents.")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="output JSON path")
    parser.add_argument(
        "--min-chars", type=int, default=MIN_TEXT_CHARS, help="drop docs whose text is shorter than this"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    docs = merge(SOURCE_FILES, Path(args.out), args.min_chars)
    sys.exit(0 if docs else 1)
