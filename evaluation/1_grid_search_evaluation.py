"""
1_grid_search_evaluation.py — grid search over keyword-search field boosts.

Usage:

uv run python evaluation/1_grid_search_evaluation.py
uv run python evaluation/1_grid_search_evaluation.py --k 5

Scores against the fixed test-question set in
evaluation/test_questions_for_eval.json (generated once, reused as-is).

Output:
    evaluation/results_1_grid_search_evaluation.json
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from my_assistant.rag import RAG, KEYWORD_BOOST, rrf

from _eval_utils import (
    CANDIDATES_PER_LEG,
    atomic_write_json,
    best_rank_of,
    load_eval_questions,
    target_doc_ids,
)

RESULTS_FILE = Path(__file__).resolve().parent / "results_1_grid_search_evaluation.json"

K = 5  # Hit Rate@K / MRR@K, same cutoff the other evaluations use

# Candidate values per field. Every field is swept except "text", which is
# held fixed at 1.0 as the anchor every other boost is relative to. This
# fixing isn't a judgment call about text mattering less — it's mathematically
# necessary: scaling every field's boost by the same constant doesn't change
# which document scores highest for a given query, so without one fixed
# anchor the search space would contain infinitely many equivalent
# combinations (e.g. {title:2,text:1} scores documents in exactly the same
# relative order as {title:4,text:2}). Fixing text=1.0 removes that
# redundancy. component and error_type used to be fixed too, but there's no
# similar mathematical reason to hold them still, so they're now swept along
# with title/summary/tags_str to check more combinations.
GRID = {
    "title": [1.0, 2.0, 3.0, 4.0],
    "summary": [1.0, 1.5, 2.0, 2.5],
    "tags_str": [0.5, 1.0, 1.5, 2.0],
    "component": [0.25, 0.5, 1.0, 1.5],
    "error_type": [0.25, 0.5, 1.0, 1.5],
}
FIXED_BOOST = {"text": 1.0}


def candidate_boosts() -> list[dict]:
    keys = list(GRID.keys())
    combos = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        boost = dict(FIXED_BOOST)
        boost.update(dict(zip(keys, values)))
        combos.append(boost)
    return combos


def precompute_vector_legs(rag: RAG, eval_questions: list[dict], k: int) -> dict[int, list[dict]]:
    """Embed + vector-search every eval question exactly once. This is the
    only network-bound step in the whole grid search — everything after
    this is pure in-memory minsearch scoring."""
    cache: dict[int, list[dict]] = {}
    for i, pair in enumerate(tqdm(eval_questions, desc="Caching vector legs", unit="q")):
        try:
            cache[i] = rag.vector_search(pair["question"], n=CANDIDATES_PER_LEG)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"  WARNING: vector search failed for {target_doc_ids(pair)!r} — {exc}")
            cache[i] = []
    return cache


def score_boost(
    rag: RAG,
    boost: dict,
    eval_questions: list[dict],
    vector_cache: dict[int, list[dict]],
    k: int,
) -> dict:
    hits = 0
    reciprocal = 0.0
    evaluated = 0
    for i, pair in enumerate(eval_questions):
        targets = target_doc_ids(pair)
        vec = vector_cache.get(i, [])
        try:
            kw = rag.keyword_index.search(
                pair["question"], boost_dict=boost, num_results=CANDIDATES_PER_LEG
            )
        except Exception:  # noqa: BLE001
            continue
        id_to_doc: dict[str, dict] = {}
        for doc in kw + vec:
            id_to_doc.setdefault(doc["id"], doc)
        fused_ids = rrf([kw, vec])
        fused = [id_to_doc[i2] for i2 in fused_ids if i2 in id_to_doc]

        evaluated += 1
        rank = best_rank_of(targets, fused, k)
        if rank:
            hits += 1
            reciprocal += 1.0 / rank

    return {
        "boost": boost,
        "hit_rate": hits / evaluated if evaluated else 0.0,
        "mrr": reciprocal / evaluated if evaluated else 0.0,
        "n_evaluated": evaluated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grid search over keyword-search field boosts "
        "(title/summary/tags_str/component/error_type; text held fixed as the anchor)."
    )
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--limit", type=int, default=None, help="use only N test questions")
    args = parser.parse_args()

    print("=" * 60)
    print("1. GRID SEARCH — keyword-search field boosts")
    print("=" * 60)

    eval_questions = load_eval_questions(limit=args.limit)
    print(f"Loaded {len(eval_questions)} test questions")

    print("Building indices (once)...")
    rag = RAG()

    combos = candidate_boosts()
    grid_desc = " x ".join(f"{field}({len(values)})" for field, values in GRID.items())
    print(f"Grid: {grid_desc} = {len(combos)} combinations, fixed: {FIXED_BOOST}")

    print("Caching vector-search leg per question (only network-bound step)...")
    vector_cache = precompute_vector_legs(rag, eval_questions, args.k)

    print(f"Scoring {len(combos)} boost combinations (pure in-memory keyword re-scoring)...")
    started = time.time()
    scored = []
    for boost in tqdm(combos, desc="Grid search", unit="combo"):
        scored.append(score_boost(rag, boost, eval_questions, vector_cache, args.k))
    elapsed = time.time() - started

    scored.sort(key=lambda r: (r["hit_rate"], r["mrr"]), reverse=True)
    winner = scored[0]

    # Score the current production boosts too, as the baseline to beat.
    current = score_boost(rag, dict(KEYWORD_BOOST), eval_questions, vector_cache, args.k)

    beats_current = winner["hit_rate"] > current["hit_rate"] or (
        winner["hit_rate"] == current["hit_rate"] and winner["mrr"] > current["mrr"]
    )

    print(f"\nDone in {elapsed:.1f}s\n")
    print(f"{'Rank':<5} | {'Hit Rate@' + str(args.k):<10} | {'MRR@' + str(args.k):<8} | boost")
    print("-" * 70)
    for i, row in enumerate(scored[:10], start=1):
        print(f"{i:<5} | {row['hit_rate']:<10.3f} | {row['mrr']:<8.3f} | {row['boost']}")

    print(f"\nCurrent production boost -> Hit Rate@{args.k}={current['hit_rate']:.3f}, "
          f"MRR@{args.k}={current['mrr']:.3f}")
    print(f"Best boost found         -> Hit Rate@{args.k}={winner['hit_rate']:.3f}, "
          f"MRR@{args.k}={winner['mrr']:.3f}")
    print(f"Best boost values: {winner['boost']}")
    if beats_current:
        print("\nThis beats the current production boost — "
              "copy them into my_assistant/rag.py's KEYWORD_BOOST.")
    else:
        print("\nThe current production boost is already at least as good as anything in this "
              "grid — no change recommended.")

    # "best_result" is deliberately the very first key so it's the first thing
    # visible when opening the results file — no need to apply anything to
    # see the headline finding.
    payload = {
        "best_result": {
            "boost": winner["boost"],
            "hit_rate": winner["hit_rate"],
            "mrr": winner["mrr"],
            "beats_current_production_boost": beats_current,
            "current_production_boost": dict(KEYWORD_BOOST),
            "current_production_result": {"hit_rate": current["hit_rate"], "mrr": current["mrr"]},
        },
        "k": args.k,
        "n_questions": len(eval_questions),
        "grid": GRID,
        "fixed_boost": FIXED_BOOST,
        "n_combinations": len(combos),
        "elapsed_seconds": round(elapsed, 2),
        "top_10": scored[:10],
        "all_results": scored,
    }
    atomic_write_json(RESULTS_FILE, payload)
    print(f"\nSaved full grid results -> {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
