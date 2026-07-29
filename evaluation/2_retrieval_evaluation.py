"""
2_retrieval_evaluation.py — retrieval quality evaluation.

This is evaluation step 2, run AFTER 1_grid_search_evaluation.py: it scores
keyword / vector / hybrid retrieval using whatever KEYWORD_BOOST is
currently live in my_assistant/rag.py (the winner of step 1, once applied).

Two complementary checks, both against the same three retrieval modes:

  A. Doc-level Hit Rate@k / MRR@k against the fixed test-question set in
     evaluation/test_questions_for_eval.json (one realistic question per
     sampled document, so we know exactly which single document is
     "correct"). This file was generated once and is reused as-is — this
     script has no way to regenerate it.

  B. Category-level Hit Rate@k against the 50 hand-written questions in
     test_questions.md — a human-written sanity check that the right KIND
     of document shows up, independent of the test-question set above. Five
     general/mixed questions are reported qualitatively only (no single
     expected category).

Usage:
    uv run python evaluation/2_retrieval_evaluation.py                    # run both checks
    uv run python evaluation/2_retrieval_evaluation.py --k 5
    uv run python evaluation/2_retrieval_evaluation.py --show-titles

Output:
    evaluation/results_2_retrieval_evaluation.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from my_assistant.rag import RAG, KEYWORD_BOOST, rrf

from _eval_utils import (
    CANDIDATES_PER_LEG,
    EVAL_QUESTIONS_FILE,
    atomic_write_json,
    best_rank_of,
    load_eval_questions,
    target_doc_ids,
)

RESULTS_FILE = Path(__file__).resolve().parent / "results_2_retrieval_evaluation.json"
K = 5
METHODS = ("keyword", "vector", "hybrid")

# question text -> expected category (must match the `category` field stored
# on documents in data/enriched_documents.json). None = general/mixed
# question with no single expected category — reported qualitatively only.
CATEGORY_QUESTIONS: list[tuple[str, str | None]] = [
    ("My Spark job keeps failing with java.lang.OutOfMemoryError: Java heap space during the "
     "shuffle stage of a large groupBy aggregation. Here's my current spark-submit command: "
     "spark-submit --class com.mycompany.etl.DailyAggregationJob --master yarn --deploy-mode cluster "
     "--driver-memory 4g --executor-memory 4g --executor-cores 4 --num-executors 10 "
     "--conf spark.sql.shuffle.partitions=200 --conf spark.dynamicAllocation.enabled=false "
     "--conf spark.serializer=org.apache.spark.serializer.KryoSerializer s3://my-bucket/jobs/daily-aggregation.jar "
     "The job processes ~500GB of data with a wide join followed by a groupBy aggregation. It ran fine on "
     "smaller datasets but started failing consistently once we scaled up to this volume. Executors are "
     "dying with OOM around the shuffle-write stage. How should I fix the memory and partition configuration?",
     "executor-oom-high-shuffle"),
    ("My driver keeps crashing with OutOfMemoryError after calling .collect() on a large DataFrame. "
     "Current command: spark-submit --driver-memory 4g --executor-memory 8g --executor-cores 4 "
     "--num-executors 20 my_job.jar. The result set is larger than expected. What should I change?",
     "driver-oom-collect-heavy"),
    ("I'm seeing heavy data skew — one task in my shuffle stage takes 45 minutes while the other 199 "
     "finish in under 2 minutes. My config is --executor-memory 8g --executor-cores 4 --num-executors 15 "
     "--conf spark.sql.shuffle.partitions=200. How do I fix this?",
     "data-skew-few-large-partitions"),
    ("My job produces thousands of tiny output files (a few KB each) after a write to S3, and downstream "
     "jobs reading this data are slow. Current config uses 500 shuffle partitions on a 2GB dataset. How "
     "should I reduce the small-files problem?",
     "small-files-problem-many-output-files"),
    ("My job has too few partitions — only 4 tasks are running even though I have --num-executors 20 "
     "--executor-cores 4 (80 total cores available). What config change fixes the underutilization?",
     "too-few-partitions-underutilized-cluster"),
    ("I'm seeing my job spill heavily to disk during shuffle, with ShuffleWriteMetrics showing large "
     "'bytes spilled.' Current config: --executor-memory 4g --executor-cores 8. What should I adjust?",
     "shuffle-spill-to-disk"),
    ("My broadcast join is timing out with 'Could not execute broadcast in 300 secs'. The smaller table "
     "is around 900MB. Current config uses default spark.sql.autoBroadcastJoinThreshold. How do I fix this?",
     "slow-broadcast-join-large-table"),
    ("I have a small lookup table (under 50MB) being joined with a huge fact table via a regular "
     "sort-merge join, and it's slow. How do I force Spark to use a broadcast join instead?",
     "missing-broadcast-hint-small-table"),
    ("My job with spark.dynamicAllocation.enabled=true keeps adding and removing executors rapidly, "
     "causing instability. What settings control this thrashing?",
     "dynamic-allocation-thrashing"),
    ("I'm seeing frequent full GC pauses (several seconds each) on my executors under heavy load. "
     "Current config: --executor-memory 16g --executor-cores 8. How do I reduce GC pressure?",
     "gc-pressure-old-gen"),
    ("My job caches a DataFrame with .cache() but I'm running low on executor memory and seeing "
     "evictions. How should I tune caching vs execution memory?",
     "excessive-caching-memory-pressure"),
    ("I'm re-computing the same DataFrame transformation multiple times across my pipeline without "
     "caching it, and it's slow. What's the right way to persist it?",
     "uncached-repeated-dataframe-reuse"),
    ("My wide transformation (multiple joins + groupBy) is causing an explosion in shuffle data volume. "
     "How do I restructure the query or config to reduce shuffle size?",
     "wide-transformation-shuffle-explosion"),
    ("My job uses .checkpoint() on a DataFrame with a very long lineage, and checkpointing itself is "
     "taking a long time. What's the tradeoff here and how should I tune it?",
     "checkpoint-overhead-long-lineage"),
    ("I'm getting FetchFailedException: Failed to connect errors during a large shuffle fetch between "
     "executors. What network/timeout settings should I adjust?",
     "network-timeout-large-shuffle-fetch"),
    ("My executors are being killed by YARN with 'Container killed by YARN for exceeding memory limits', "
     "even though --executor-memory looks sufficient. What's missing from my config?",
     "executor-loss-yarn-container-killed"),
    ("My groupBy on a skewed key (one key has 80% of the rows) is creating one massive task. How do I "
     "handle this specific skew pattern?",
     "skewed-join-key-distribution"),
    ("My job's driver needs a lot of memory to plan a complex query with many joins, and I'm seeing "
     "driver OOM before any executor work starts. What driver-side settings help?",
     "insufficient-driver-memory-large-plan"),
    ("I have idle executors sitting around doing nothing most of the time, wasting cluster resources. "
     "Current config: --num-executors 50 --executor-cores 2 --executor-memory 4g. How do I right-size this?",
     "over-provisioned-executors-idle-cluster"),
    ("My job is slow because parallelism is too low relative to available cores — only using "
     "--executor-cores 1 --num-executors 5 on a large cluster. What should I change?",
     "under-provisioned-cores-slow-parallelism"),
    ("I haven't enabled off-heap memory and I'm seeing heap exhaustion during heavy shuffle. How do I "
     "configure off-heap memory correctly?",
     "off-heap-memory-not-configured"),
    ("I forgot to enable Kryo serialization and I suspect serialization overhead is slowing down my "
     "job. How do I enable it and what else should I check?",
     "kryo-serialization-not-enabled"),
    ("My job doesn't have Adaptive Query Execution (AQE) enabled and I'm seeing poor join strategy "
     "choices and static partition counts. How do I turn AQE on and what does it help with?",
     "adaptive-query-execution-disabled"),
    ("My daily batch Spark job normally runs in 20 minutes but is now taking 90 minutes without failing "
     "outright. Downstream reports are blocked. How do I diagnose this?",
     "data-pipeline-failures"),
    ("A Spark job that used to complete reliably now intermittently fails with executor loss, but "
     "there's no obvious error in the driver logs. Where do I start investigating?",
     "data-pipeline-failures"),
    ("My ETL pipeline's runtime has been growing steadily over the past few weeks even though data "
     "volume looks stable. What could cause this kind of gradual degradation?",
     "performance-latency"),
    ("I have a job that reads from a partitioned S3 dataset but seems to be scanning far more data than "
     "expected. How do I check if partition pruning is working?",
     "performance-latency"),
    ("My Kafka consumer is falling behind and lag keeps growing even though I haven't changed anything. "
     "What should I check first?",
     "streaming-kafka"),
    ("My Spark Structured Streaming job is experiencing frequent micro-batch delays and increasing "
     "backpressure. How do I diagnose the bottleneck?",
     "streaming-kafka"),
    ("I'm seeing duplicate records downstream from my Kafka-based streaming pipeline. What are the "
     "common causes and how do I fix exactly-once semantics?",
     "streaming-kafka"),
    ("Our data quality checks started failing overnight — null rates on a key column spiked from under "
     "1% to over 30%. How do I trace this back to the source?",
     "data-quality"),
    ("I'm seeing row counts in a downstream table that don't match the expected counts from the source "
     "table after a join. What's the systematic way to debug this?",
     "data-quality"),
    ("A schema validation step is rejecting records that used to pass. How do I figure out what changed "
     "upstream?",
     "data-quality"),
    ("Our monthly cloud compute costs jumped significantly after a recent pipeline change, even though "
     "data volume didn't grow much. How do I find what's driving the cost increase?",
     "cloud-cost-resources"),
    ("I have several long-running clusters that seem to be underutilized most of the time. What's a "
     "good strategy to reduce idle cluster cost?",
     "cloud-cost-resources"),
    ("My Airflow DAG has started missing its SLA and tasks are queuing up behind each other. How do I "
     "identify the bottleneck task?",
     "orchestration-scheduling"),
    ("A scheduled job occasionally fails to trigger at all, with no error logged. What are common "
     "causes of silently missed schedules?",
     "orchestration-scheduling"),
    ("A producer team changed a field type in our upstream data without notice, and it's now breaking "
     "our downstream schema. How should we have caught this earlier, and how do we handle it now?",
     "schema-evolution"),
    ("We need to add a new nullable column to an existing Delta table without breaking existing "
     "readers. What's the safe way to do this?",
     "schema-evolution"),
    ("We need to backfill three months of historical data through our pipeline, but running it all at "
     "once risks overwhelming downstream systems. What's a safe backfill strategy?",
     "backfills-reprocessing"),
    ("A bug in our transformation logic corrupted two weeks of data before we caught it. What's the "
     "process for identifying affected records and safely reprocessing them?",
     "backfills-reprocessing"),
    ("A teammate lost access to a cluster after an IAM policy update, but we're not sure exactly which "
     "permission is missing. How do we debug access issues systematically?",
     "security-access"),
    ("We need to rotate a secret used by multiple Spark jobs without causing downtime. What's the safe "
     "rotation process?",
     "security-access"),
    ("Leadership wants an ETA on a data quality fix, but the root cause isn't fully understood yet. How "
     "do I communicate this without overpromising?",
     "stakeholder-process"),
    ("Two teams disagree on whose pipeline introduced a data inconsistency, and both point to the "
     "other. How do I approach resolving this productively?",
     "stakeholder-process"),
    ("What's the difference between narrow and wide transformations in Spark, and why does it matter "
     "for performance?", None),
    ("How do I decide the right number of shuffle partitions for a given dataset size?", None),
    ("My Spark job works fine locally but fails only in the cluster environment. What's different that "
     "I should check?", None),
    ("What's a reasonable executor-to-core-to-memory ratio to start with for a general-purpose Spark "
     "job?", None),
    ("How do I tell whether a slow Spark job is CPU-bound, memory-bound, or I/O-bound?", None),
]


# ------------------------------------------------- A. doc-level (eval questions) ----


def evaluate_eval_questions(rag: RAG, eval_questions: list[dict], k: int) -> dict:
    print(f"\n[A] Doc-level Hit Rate@{k} / MRR@{k} against {len(eval_questions)} test questions")
    hits = {m: 0 for m in METHODS}
    reciprocal = {m: 0.0 for m in METHODS}
    per_question: list[dict] = []
    evaluated = 0
    failures = 0

    for pair in tqdm(eval_questions, desc="Eval questions", unit="q"):
        question, targets = pair["question"], target_doc_ids(pair)
        try:
            kw = rag.keyword_search(question, n=CANDIDATES_PER_LEG)
            vec = rag.vector_search(question, n=CANDIDATES_PER_LEG)
            fused_ids = rrf([kw, vec])
            id_to_doc: dict[str, dict] = {}
            for doc in kw + vec:
                id_to_doc.setdefault(doc["id"], doc)
            hybrid = [id_to_doc[i] for i in fused_ids if i in id_to_doc]
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"  WARNING: skipping {targets!r} — {str(exc)[:200]}")
            failures += 1
            continue

        evaluated += 1
        ranked = {"keyword": kw, "vector": vec, "hybrid": hybrid}
        row: dict[str, Any] = {"question": question, "doc_ids": targets}
        for method in METHODS:
            rank = best_rank_of(targets, ranked[method], k)
            row[f"{method}_rank"] = rank
            if rank:
                hits[method] += 1
                reciprocal[method] += 1.0 / rank
        per_question.append(row)

    metrics = {
        method: {
            "hit_rate": hits[method] / evaluated if evaluated else 0.0,
            "mrr": reciprocal[method] / evaluated if evaluated else 0.0,
            "hits": hits[method],
        }
        for method in METHODS
    }

    label = {"keyword": "Keyword", "vector": "Vector", "hybrid": "Hybrid"}
    print(f"\n{'Method':<9} | {'Hit Rate@' + str(k):<10} | MRR@{k}")
    print(f"{'-' * 9}-+-{'-' * 10}-+-{'-' * 7}")
    for method in METHODS:
        m = metrics[method]
        print(f"{label[method]:<9} | {m['hit_rate']:<10.2f} | {m['mrr']:.2f}")

    best = max(METHODS, key=lambda m: metrics[m]["hit_rate"])
    print(f"Best method by Hit Rate@{k}: {label[best]} ({metrics[best]['hit_rate']:.2f})")

    return {
        "n_questions": evaluated,
        "n_failed": failures,
        "metrics": metrics,
        "best_method": best,
        "per_question": per_question,
    }


# ------------------------------------------------- B. category-level (human) ----


def evaluate_categories(rag: RAG, mode: str, k: int) -> dict:
    scored = [q for q in CATEGORY_QUESTIONS if q[1] is not None]
    unscored = [q for q in CATEGORY_QUESTIONS if q[1] is None]

    def search(question: str) -> list[dict]:
        if mode == "keyword":
            return rag.keyword_search(question, n=k)
        if mode == "vector":
            return rag.vector_search(question, n=k)
        return rag.hybrid_search(question, n=k)

    hits = 0
    per_question = []
    for question, expected_category in scored:
        docs = search(question)
        categories = [d.get("category", "") for d in docs]
        titles = [d.get("title", "") for d in docs]
        hit = expected_category in categories
        hits += int(hit)
        per_question.append(
            {
                "question": question[:100],
                "expected_category": expected_category,
                "hit": hit,
                "retrieved_categories": categories,
                "retrieved_titles": titles,
            }
        )

    qualitative = []
    for question, _ in unscored:
        docs = search(question)
        qualitative.append(
            {"question": question[:100], "retrieved_titles": [d.get("title", "") for d in docs]}
        )

    return {
        "mode": mode,
        "n_scored": len(scored),
        "hits": hits,
        "hit_rate_at_k": round(hits / len(scored), 3) if scored else 0.0,
        "per_question": per_question,
        "qualitative_general": qualitative,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval evaluation: doc-level eval questions + category-level human questions."
    )
    parser.add_argument("--k", type=int, default=K, help="cutoff for Hit Rate / MRR")
    parser.add_argument("--limit", type=int, default=None, help="use only N eval questions")
    parser.add_argument(
        "--category-mode", choices=["keyword", "vector", "hybrid", "all"], default="all",
        help="which retrieval mode(s) to run the category-level check with"
    )
    parser.add_argument("--show-titles", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("2. RETRIEVAL EVALUATION")
    print("=" * 60)

    eval_questions = load_eval_questions(limit=args.limit)
    print(f"Loaded {len(eval_questions)} test questions from {EVAL_QUESTIONS_FILE}")
    print(f"Live KEYWORD_BOOST: {KEYWORD_BOOST}")

    print("Building indices...")
    rag = RAG()
    started = time.time()

    part_a = evaluate_eval_questions(rag, eval_questions, args.k)

    print(f"\n[B] Category-level Hit Rate@{args.k} against {len(CATEGORY_QUESTIONS)} hand-written questions")
    category_modes = ["keyword", "vector", "hybrid"] if args.category_mode == "all" else [args.category_mode]
    part_b = {}
    print(f"\n{'Mode':<10} | {'Category Hit Rate@' + str(args.k):<20} | Hits/Total")
    print("-" * 55)
    for mode in category_modes:
        result = evaluate_categories(rag, mode, k=args.k)
        part_b[mode] = result
        print(f"{mode:<10} | {result['hit_rate_at_k']:<20} | {result['hits']}/{result['n_scored']}")

    if args.show_titles:
        for mode in category_modes:
            print(f"\n--- {mode} misses ---")
            for pq in part_b[mode]["per_question"]:
                if not pq["hit"]:
                    print(f"  Q: {pq['question']}")
                    print(f"    expected: {pq['expected_category']}")
                    print(f"    got: {pq['retrieved_categories']}")

    elapsed = time.time() - started
    payload = {
        "k": args.k,
        "keyword_boost_used": dict(KEYWORD_BOOST),
        "elapsed_seconds": round(elapsed, 2),
        "eval_questions_evaluation": part_a,
        "category_evaluation": part_b,
    }
    atomic_write_json(RESULTS_FILE, payload)
    print(f"\nSaved detailed results -> {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
