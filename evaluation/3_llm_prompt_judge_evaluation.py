"""
3_llm_prompt_judge_evaluation.py — LLM-as-a-judge evaluation of answer-prompt variants.

This is evaluation step 3, run after retrieval has already been decided by
step 1 (boost grid search) and after step 2 (retrieval evaluation) has
confirmed search itself is working — it answers a different question: given
that retrieval is settled, which ANSWER PROMPT produces better answers?

Takes N questions from the fixed evaluation/test_questions_for_eval.json set
(generated once, reused as-is — no regeneration path), runs each through the full RAG
pipeline once per prompt variant below, then asks a judge LLM to score each
answer 1-5 on:

  * groundedness — is the answer supported by the retrieved documents?
  * relevance    — does it actually answer the question?

Four meaningfully different prompt variants/personas are compared, not just
wording tweaks on the same style:

  1. sme_concise        — big-data subject matter expert: crisp, authoritative,
                           a fixed fix with no hedging or surrounding prose.
  2. de_examples         — Spark/DE expert: explains the reasoning and includes
                           a concrete worked example grounded in the context.
  3. error_resolution    — error-resolution specialist: solution only, as few
                           words as possible, optimized to copy-paste and go.
  4. structured_current  — the structured "Likely cause -> steps -> verify ->
                           caveats" prompt that is this project's live default
                           in my_assistant/rag.py's ANSWER_PROMPT.

Prints a comparison table and writes evaluation/results_3_llm_prompt_judge_evaluation.json.

The prompt swap is done with a context manager that temporarily rebinds
`rag.ANSWER_PROMPT` and always restores it, so rag.py's default behaviour is
never permanently changed and no retrieval/generation logic is duplicated.

Running all 4 variants means 4x the LLM calls of a single-prompt run — use
--n to control cost/time (each question runs through all 4 variants, plus
one judge call per answer, so total LLM calls ~= n * 4 * 2).

Usage:
    uv run python evaluation/3_llm_prompt_judge_evaluation.py
    uv run python evaluation/3_llm_prompt_judge_evaluation.py --force    # ignore cached results
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import my_assistant.rag as rag_module
from my_assistant.rag import RAG, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, OPENAI_BASE_URL, OPENAI_MODEL, format_context

from _eval_utils import (
    EVAL_QUESTIONS_FILE,
    LLM_MODEL,
    atomic_write_json,
    extract_json,
    load_eval_questions,
    target_doc_ids,
    make_llm_client,
    no_think_kwargs,
)

RESULTS_FILE = Path(__file__).resolve().parent / "results_3_llm_prompt_judge_evaluation.json"

N_QUESTIONS = 30
CHECKPOINT_EVERY = 5
JUDGE_MAX_TOKENS = 600

# ---- variant 1: big-data SME — crisp, concise, a fixed solution ------------
PROMPT_SME_CONCISE = """\
You are a big data subject matter expert. A colleague has hit a production failure and needs the
fix, not a lecture. You have seen this class of problem many times; you state the fix plainly and
move on.

Grounding rules (absolute):
- Use ONLY the context documents below. Every config key, parameter value, command, error code,
  API, and file path you write must appear verbatim in the context. Invent nothing.
- If the context does not cover the question, reply with exactly one line:
  "The retrieved documents do not cover this issue." Then stop. Do not guess or generalize.

Style rules:
- Be terse and authoritative. State the fix as fact, not as a possibility.
- Banned: "might", "could", "you may want to", "it depends", "generally", "I would suggest",
  "hope this helps", and any restating of the question.
- Do not explain the underlying theory, do not describe how the system works internally, and do not
  add background. Only what to change and what to run.
- Do not add headings, preambles, or closing remarks beyond the shape below.

Output shape — follow it exactly:

Cause: <one sentence, max 20 words, naming the root cause.>

Fix:
- <imperative action with the exact config key and value, or the exact command, from the context>
- <second action, if genuinely required>
- <third action, if genuinely required — never more than four bullets total>

If a config change is the fix, show it as a single code block of the exact keys and values, nothing
else. Total answer must be under 120 words.

Question: {question}

Context documents:
{context}

Answer:"""

# ---- variant 2: Spark/DE expert — explains the reasoning with an example ----
PROMPT_DE_EXAMPLES = """\
You are a Spark and data engineering expert who teaches by example. Your reader is a competent
engineer who wants to understand why the fix works, so that they recognize the pattern the next
time it appears — not just paste a command.

Grounding rules (absolute):
- Everything you assert must be supported by the context documents below: config keys, default and
  recommended values, commands, error codes, symptoms, component names. Never introduce a knob,
  flag, metric, or error string that is not in the context.
- Your worked example must be built only from values, symptoms, and settings that appear in the
  context. If the context gives no concrete numbers, use the qualitative before/after that the
  context does describe, and say plainly that the context gives no specific values — do not fabricate
  plausible-looking numbers.
- If the context does not cover the question, say so directly, name what is missing, and stop. Do not
  fill gaps from general knowledge.

Write the answer as flowing technical prose in three parts, with these exact headings:

**What's actually happening**
Two to four sentences tracing the failure from the symptom back to the mechanism, in the order a
reader would reason about it. Connect the observed error to the underlying behavior described in the
context.

**The fix, and why it works**
Explain the change and the causal chain: this setting controls X, so raising/lowering it makes Y
happen, which removes the failure. Name the exact keys and values from the context inline.

**Worked example**
One concrete illustration grounded in the context — a before/after config snippet in a code block, a
short numbered walk-through of what the values change, or a small scenario showing the difference in
behavior. Make the contrast explicit: what the run looked like before, what it looks like after.

Aim for 200-350 words. Prefer one well-developed example over several shallow ones. No bulleted
summary at the end; end on the example.

Question: {question}

Context documents:
{context}

Answer:"""

# ---- variant 3: error-resolution specialist — solution only, ultra terse ----
PROMPT_ERROR_RESOLUTION = """\
You are a data engineer answering in an incident channel. The reader wants the solution and nothing
else. They will copy it, apply it, and close the tab.

Rules:
- Output the solution only. No greeting, no diagnosis, no explanation, no rationale, no caveats, no
  verification steps, no closing line, no headings, no restating the question.
- Every config key, value, command, flag, and code path must come verbatim from the context
  documents below. Invent nothing.
- Prefer a code block. If the fix is a config change or a command, output the code block and nothing
  around it.
- If the fix genuinely cannot be expressed as code, output at most two imperative sentences.
- If more than one step is required, output them as a bare numbered list, one short imperative line
  each, maximum four lines.
- Hard cap: 60 words of prose outside code blocks.
- If the context does not contain the fix, output exactly: Not covered by the retrieved documents.

Question: {question}

Context documents:
{context}

Answer:"""

# ---- variant 4: structured — this project's current production default ----
# Kept as a literal copy (not a dynamic reference to rag_module.ANSWER_PROMPT)
# so this comparison stays meaningful even if rag.py's default is changed
# later — this variant always represents "the structured prompt as it stood
# when this evaluation script was written," not whatever happens to be live.
PROMPT_STRUCTURED = """\
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

VARIANTS: dict[str, dict[str, str]] = {
    "1_sme_concise": {"label": "1: SME (concise, fixed fix)", "template": PROMPT_SME_CONCISE},
    "2_de_examples": {"label": "2: DE expert (explains with example)", "template": PROMPT_DE_EXAMPLES},
    "3_error_resolution": {"label": "3: Error-resolution (terse, solution-only)", "template": PROMPT_ERROR_RESOLUTION},
    "4_structured_current": {"label": "4: Structured (current production default)", "template": PROMPT_STRUCTURED},
}

JUDGE_PROMPT = """\
You are a strict evaluator of a retrieval-augmented data engineering assistant.

Score the ANSWER on two dimensions, each an integer from 1 to 5:

groundedness — is every claim in the answer supported by the CONTEXT DOCUMENTS?
  5 = fully supported, no invented commands/configs/error codes
  3 = mostly supported, some unsupported but plausible detail
  1 = largely hallucinated or contradicts the context

relevance — does the answer address the USER QUESTION?
  5 = directly and completely answers it, actionable
  3 = partially answers it or is padded with off-topic material
  1 = does not answer the question

Return ONLY valid JSON, no markdown fences, no explanation outside the JSON:
{{"groundedness": N, "relevance": N, "reasoning": "one or two sentences"}}

USER QUESTION:
{question}

CONTEXT DOCUMENTS:
{context}

ANSWER:
{answer}
"""


@contextlib.contextmanager
def answer_prompt(template: str):
    """Temporarily swap rag.ANSWER_PROMPT; always restore it."""
    original = rag_module.ANSWER_PROMPT
    rag_module.ANSWER_PROMPT = template
    try:
        yield
    finally:
        rag_module.ANSWER_PROMPT = original


@contextlib.contextmanager
def quiet():
    """Swallow rag.py's [1/7]... step logging so tqdm output stays readable."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


def load_questions(n: int) -> list[dict]:
    return load_eval_questions(limit=n)


def answer_with_variant(rag: RAG, question: str, template: str) -> dict | None:
    """Full RAG pipeline under a given answer prompt. Retries once."""
    for attempt in (1, 2):
        try:
            with answer_prompt(template), quiet():
                # self_check off — evaluating the raw prompt-A-vs-prompt-B
                # difference; a self-check/repair pass would blur that comparison
                result = rag.query(question, return_docs=True, self_check=False)
            return result  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                tqdm.write(f"  WARNING: answer failed — {str(exc)[:200]}")
            else:
                time.sleep(1.5)
    return None


def judge(client: OpenAI, question: str, answer: str, docs: list[dict]) -> dict | None:
    """Score one answer. Retries once, then returns None (item is skipped)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=format_context(docs)[:12000],
        answer=answer[:6000],
    )
    last = ""
    for attempt in (1, 2):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=JUDGE_MAX_TOKENS,
                **no_think_kwargs(),
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = extract_json(raw)
            if parsed and "groundedness" in parsed and "relevance" in parsed:
                return {
                    "groundedness": clamp_score(parsed.get("groundedness")),
                    "relevance": clamp_score(parsed.get("relevance")),
                    "reasoning": str(parsed.get("reasoning", ""))[:600],
                }
            last = f"unparseable judge response: {raw[:160]!r}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:200]
        if attempt == 1:
            time.sleep(1.5)
    tqdm.write(f"  WARNING: judge failed — {last}")
    return None


def clamp_score(value: Any) -> int | None:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(1, min(5, score))


def evaluate_llm(
    n: int = N_QUESTIONS,
    out_path: Path = RESULTS_FILE,
    llm_base_url: str | None = None,
    llm_model: str | None = None,
) -> dict:
    print("=" * 60)
    print("3. LLM-AS-JUDGE EVALUATION")
    print("=" * 60)

    questions = load_questions(n)
    print(f"[1/4] Loaded {len(questions)} questions from {EVAL_QUESTIONS_FILE}")

    print("[2/4] Building indices...")
    with quiet():
        rag = RAG(llm_base_url=llm_base_url, llm_model=llm_model) if llm_base_url else RAG()
    print(f"      LLM: {rag.llm_model} @ {rag.llm_base_url} (provider={rag.llm_provider})")
    judge_client = make_llm_client()

    print(f"[3/4] Running {len(questions)} questions x {len(VARIANTS)} prompt variants...")
    rows: list[dict] = []
    failures = 0
    started = time.time()

    total = len(questions) * len(VARIANTS)
    bar = tqdm(total=total, desc="Answer + judge", unit="run")
    try:
        for pair in questions:
            question = pair["question"]
            for key, variant in VARIANTS.items():
                result = answer_with_variant(rag, question, variant["template"])
                if result is None:
                    failures += 1
                    bar.update(1)
                    continue

                answer, docs = result["answer"], result["docs"]
                scores = judge(judge_client, question, answer, docs)
                if scores is None:
                    failures += 1
                    bar.update(1)
                    continue

                rows.append(
                    {
                        "variant": key,
                        "question": question,
                        "expected_doc_ids": target_doc_ids(pair),
                        "retrieved_doc_ids": [d.get("id") for d in docs],
                        "answer": answer,
                        "groundedness": scores["groundedness"],
                        "relevance": scores["relevance"],
                        "reasoning": scores["reasoning"],
                        "total_tokens": result["trace"].get("total_tokens"),
                        "estimated_cost_usd": result["trace"].get("estimated_cost_usd"),
                        "total_ms": result["trace"].get("total_ms"),
                    }
                )
                bar.update(1)
                if len(rows) % CHECKPOINT_EVERY == 0:
                    atomic_write_json(out_path, {"partial": True, "results": rows})
                    bar.set_postfix_str(f"checkpoint @ {len(rows)}")
    except KeyboardInterrupt:
        bar.close()
        print("\nInterrupted — saving partial results...")
        atomic_write_json(out_path, {"partial": True, "results": rows})
        raise SystemExit(130)
    finally:
        bar.close()

    if not rows:
        raise SystemExit("No answers were scored — are the LLM/embedding servers running?")

    summary: dict[str, dict] = {}
    for key, variant in VARIANTS.items():
        subset = [r for r in rows if r["variant"] == key]
        grounded = [r["groundedness"] for r in subset if r["groundedness"] is not None]
        relevant = [r["relevance"] for r in subset if r["relevance"] is not None]
        summary[key] = {
            "label": variant["label"],
            "n": len(subset),
            "avg_groundedness": (sum(grounded) / len(grounded)) if grounded else 0.0,
            "avg_relevance": (sum(relevant) / len(relevant)) if relevant else 0.0,
            "avg_total_tokens": (
                sum(r["total_tokens"] or 0 for r in subset) / len(subset) if subset else 0.0
            ),
            "total_cost_usd": sum(r["estimated_cost_usd"] or 0.0 for r in subset),
        }

    elapsed = time.time() - started
    print(f"\n[4/4] Results — {len(rows)} scored answers, {elapsed:.1f}s\n")
    print(f"{'Variant':<20} | {'Groundedness':<12} | {'Relevance':<9} | N")
    print(f"{'-' * 20}-+-{'-' * 12}-+-{'-' * 9}-+---")
    for key in VARIANTS:
        s = summary[key]
        print(
            f"{s['label']:<20} | {s['avg_groundedness']:<12.2f} | "
            f"{s['avg_relevance']:<9.2f} | {s['n']}"
        )

    best = max(
        summary,
        key=lambda k: (summary[k]["avg_groundedness"] + summary[k]["avg_relevance"]),
    )
    print(f"\nBest variant (groundedness + relevance): {summary[best]['label']}")
    for key in VARIANTS:
        s = summary[key]
        print(
            f"  {s['label']}: avg {s['avg_total_tokens']:.0f} tokens/query, "
            f"${s['total_cost_usd']:.4f} total"
        )

    payload = {
        "partial": False,
        "n_questions": len(questions),
        "n_scored": len(rows),
        "n_failed": failures,
        "elapsed_seconds": round(elapsed, 2),
        "variants": {k: {"label": v["label"], "template": v["template"]} for k, v in VARIANTS.items()},
        "summary": summary,
        "best_variant": best,
        "results": rows,
    }
    atomic_write_json(out_path, payload)
    print(f"\nSaved detailed results -> {out_path}")
    if failures:
        print(f"Skipped {failures} runs after LLM failures")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge comparison of 4 answer prompt variants/personas."
    )
    parser.add_argument("--n", type=int, default=N_QUESTIONS, help="questions to evaluate")
    parser.add_argument(
        "--force", action="store_true", help="ignore an existing results file and rerun"
    )
    parser.add_argument("--deepseek", action="store_true", help="use DeepSeek instead of the default provider")
    parser.add_argument("--openai", action="store_true", help="use OpenAI (ChatGPT) instead of the default provider")
    args = parser.parse_args()

    if args.deepseek and args.openai:
        print("Pass only one of --deepseek / --openai")
        return 1
    llm_base_url = llm_model = None
    if args.deepseek:
        llm_base_url, llm_model = DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    elif args.openai:
        llm_base_url, llm_model = OPENAI_BASE_URL, OPENAI_MODEL

    if RESULTS_FILE.exists() and not args.force:
        try:
            cached = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = {"partial": True}
        if not cached.get("partial") and cached.get("results"):
            print(f"Cached results found at {RESULTS_FILE} ({cached.get('n_scored')} scored).")
            print("Pass --force to rerun.\n")
            print(f"{'Variant':<20} | {'Groundedness':<12} | {'Relevance':<9} | N")
            print(f"{'-' * 20}-+-{'-' * 12}-+-{'-' * 9}-+---")
            for s in cached.get("summary", {}).values():
                print(
                    f"{s['label']:<20} | {s['avg_groundedness']:<12.2f} | "
                    f"{s['avg_relevance']:<9.2f} | {s['n']}"
                )
            return 0

    evaluate_llm(n=args.n, llm_base_url=llm_base_url, llm_model=llm_model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
