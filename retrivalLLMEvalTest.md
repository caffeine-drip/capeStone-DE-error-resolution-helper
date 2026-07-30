# Retrieval & LLM Evaluation

This document explains how well the DE IncidentIQ actually works, and how that was
measured. It's kept separate from [`README.md`](README.md) so the main document can stay focused
on what the app does and how to run it.

**Why does testing matter here?** Anyone can claim a search tool "works well." The only way to
back that up is to test it against questions where you already know what the right answer should
be, count how often it actually finds the right answer, and be upfront about the parts of that
testing process that are still weak. That's what this document and its underlying test scripts do.

## Running all three tests yourself

Before running any of these, make sure a model is available: either start the two local model
servers described in [`nonDockerRun.md`](nonDockerRun.md) (answer model + `Qwen3-Embedding-0.6B`),
or set `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` as a real environment variable if you'd rather use a
paid cloud provider for the steps that write answers. Run all commands from the project's main
folder.

Steps 1, 2, and 3 below all read the same fixed test-question set from
`evaluation/test_questions_for_eval.json` (103 questions — 80 generated once from sampled write-ups,
plus 23 merged in from the Spark config-tuning questions in `test_questions.md`). This set is
reused as-is going forward — there's no regeneration command, so results stay comparable across
runs.

**1. Step 1 — tune the search weights** (fast, no AI model needed for this one — pure in-memory
scoring):
```powershell
uv run python evaluation/1_grid_search_evaluation.py
```
This only measures and reports — it never changes any file. Open
`evaluation/results_1_grid_search_evaluation.json` and look at `"best_result"` at the very top: it
shows the best boost combination found, its Hit Rate/MRR, and whether it actually beats the current
production boost. If it does, copy the values by hand into `KEYWORD_BOOST` in `my_assistant/rag.py`
— see "Applying a new set of boost values" below for the exact steps.

**2. Step 2 — measure search accuracy**, now scored against whatever weights step 1 left in place:
```powershell
uv run python evaluation/2_retrieval_evaluation.py
```

**3. Step 3 — compare the 4 answer-writing personas.** This one is the most expensive (4 personas
x every question x 2 AI calls each) — smoke-test first, then run in full:
```powershell
uv run python evaluation/3_llm_prompt_judge_evaluation.py --n 10
uv run python evaluation/3_llm_prompt_judge_evaluation.py --force
```
(`--force` is needed for the full run since the smoke test already saved a results file.)

**Using a specific AI provider for a single run:** step 3 normally follows whatever `LLM_PROVIDER`
is set to in `.env`. Override that for just one run with:
```powershell
uv run python evaluation/3_llm_prompt_judge_evaluation.py --deepseek
uv run python evaluation/3_llm_prompt_judge_evaluation.py --openai
```
(Steps 1 and 2 never write an answer, so there's no provider flag for them — only the embedding
server matters there.)

Each script saves its own `evaluation/results_N_*.json` file. Nothing here is required to just use
the app day-to-day — this is purely for measuring and improving it.

## How the system is tested, in three steps

Testing is split into three separate, numbered scripts inside the `evaluation/` folder, run in
order, each one answering a different question and saving its own results file
(`evaluation/results_N_<name>.json`). Later steps build on decisions made by earlier ones — for
example, step 2 measures search accuracy using whatever search-weight settings step 1 decided were
best.

### Step 1 — tuning the search weights (`1_grid_search_evaluation.py`)

**What this checks:** As explained in the main README, the exact-word search tool gives more
weight to a match in a write-up's title than to a match buried in the body text — on the reasoning
that a title match is a much stronger sign of relevance. But how much more weight is "right"? This
step answers that with measurement instead of guesswork: it automatically tries many different
combinations of weight values, checks each one against real test questions, and reports which
combination finds the correct write-up most often. This is the same kind of systematic "try many
combinations and keep the winner" approach (called a grid search) that the LLM Zoomcamp course
itself uses for tuning its own search weights.

This step is cheap to run repeatedly: trying a different weight combination doesn't require asking
an AI model anything, so a full sweep over roughly 1,000 combinations finishes in a few minutes
(the actual 1,024-combination run took about 6.3 minutes) rather than the hours a full LLM-based
evaluation would take.

Run it with:
```powershell
uv run python evaluation/1_grid_search_evaluation.py
```
This prints the top-scoring combinations and saves the full comparison to
`evaluation/results_1_grid_search_evaluation.json`, with the single best combination called out at
the top of that file under `"best_result"`. The script never edits any file itself — applying a
winning combination to `my_assistant/rag.py` is a manual step, on purpose, so nothing changes
production behavior without a human looking at the numbers first.

**Applying a new set of boost values.** If `results_1_grid_search_evaluation.json`'s
`"best_result"` shows `"beats_current_production_boost": true`, apply it by hand:

1. Open `evaluation/results_1_grid_search_evaluation.json` and copy the `"boost"` object under
   `"best_result"` (it has `title`, `summary`, `tags_str`, `text`, `component`, `error_type`).
2. Open `my_assistant/rag.py` and find the `KEYWORD_BOOST` dict near the top of the file.
3. Replace each value in `KEYWORD_BOOST` with the matching value from the `"boost"` object you
   copied — same six keys, just new numbers.
4. Save the file. No other code changes are needed; every script reads `KEYWORD_BOOST` from
   `my_assistant/rag.py` directly, so steps 2 and 3 will automatically be scored against the new
   values the next time they're run.
5. Re-run `evaluation/2_retrieval_evaluation.py` to confirm the new values actually improved
   real search results, not just the grid-search's own internal scoring.

**Status: this step has been run.** All 5 non-anchor fields (`title`, `summary`, `tags_str`,
`component`, `error_type`) were swept — 1,024 combinations in total, scored against all 103
questions at Hit Rate@5 / MRR@5, in 377 seconds (no LLM calls, pure in-memory keyword re-scoring).

The winning combination found by the grid search is:

| Field | Winning value |
|---|---|
| `title` | 1.0 |
| `summary` | 2.0 |
| `tags_str` | 1.5 |
| `component` | 1.0 |
| `error_type` | 0.25 |
| `text` (fixed anchor) | 1.0 |

**This is identical to the boost values already live in `my_assistant/rag.py`** — they were set by
hand before this wider sweep was run, and the sweep independently confirms they're already the best
combination out of all 1,024 tried (`beats_current_production_boost: false`, because the "winner"
and "current production" are the same combination, tied at Hit Rate@5 = 0.990 / MRR@5 = 0.944). So
no further change is needed: the current `KEYWORD_BOOST` values are a measured result, not just
reasoned judgment.

### Step 2 — how good is the search? (`2_retrieval_evaluation.py`)

**What this checks:** Given the search weights from step 1, how often does the system actually
find the right write-up for a given question? This is tested two different ways, using the
production `KEYWORD_BOOST` values confirmed by step 1 above.

**A. Against the fixed 103-question eval set.** This is doc-level scoring: for each question, is
the one (or, for the 23 merged config-tuning questions, either of the two) correct write-up in the
top 5 results?

| Search style | Found the right answer (Hit Rate@5) | Ranked it near the top (MRR@5) |
|---|---|---|
| Exact-word only | 0.951 (98/103) | 0.907 |
| Meaning-based only | 0.961 (99/103) | 0.916 |
| Combined (both at once) | **0.990 (102/103)** | **0.944** |

**B. Against real, hand-written questions.** A second, independent check used 45 scoreable
questions written by a person by hand, including full, realistic error messages and commands.
Success meant a write-up from the expected category showed up in the top 5. For 23 of these 45
(the Spark config-tuning scenarios), "category" means a specific scenario slug that only 2
near-duplicate write-ups share — close to an exact-document check. For the other 22 (broader
incident types like "data quality" or "orchestration/scheduling"), the category spans 5-15
write-ups, so it's a genuinely looser check than "the exact one write-up."

| Search style | Found the right category (Hit Rate@5) |
|---|---|
| Exact-word only | 0.844 (38 of 45) |
| Meaning-based only | 0.889 (40 of 45) |
| Combined (both at once) | **0.956 (43 of 45)** |

**Both checks agree: combined search wins, and by a clearer margin than earlier runs suggested.**
The human-written questions (part B) are the harder, more realistic test — they contain exact
technical phrases (an exact setting name, a full pasted command) that meaning-based search alone
can underweight, and combined search catches both that and the meaning-based matches. On both the
103-question set and the 45 hand-written questions, hybrid search comes out ahead of either single
method alone. **The app uses the combined approach as a result — now backed by both test sets
pointing the same direction, not just the human-written one.**

```powershell
uv run python evaluation/2_retrieval_evaluation.py                # run both checks above
```

### Step 3 — which wording of the answer works better? (`3_llm_prompt_judge_evaluation.py`)

**What this checks:** Once search is settled, does *how the answer is written* matter? Four
meaningfully different personas/styles of instructions for writing the answer are compared, not
just small wording tweaks on the same style:

| Persona | What it optimizes for |
|---|---|
| 1. Big-data subject matter expert | crisp, authoritative, a fixed fix with no hedging or surrounding prose |
| 2. Spark/DE expert with examples | explains the reasoning and includes a concrete worked example |
| 3. Error-resolution specialist | solution only, as few words as possible, copy-paste and go |
| 4. Structured (current default) | likely cause -> numbered steps -> how to verify -> caveats |

Each question is run through all 4 personas, and a second AI model, acting as an independent judge,
scores every answer 1 to 5 on two things: whether every claim in the answer was actually backed up
by the retrieved write-ups (not invented), and whether the answer actually addressed the question
asked.

**Status: this comparison has been run across all 4 personas.** 30 questions x 4 personas x 2 AI
calls each (one to write the answer, one to judge it) = 120 scored answers, no failures.

| Persona | Backed up by facts (1-5) | Answered the question (1-5) | Average tokens/answer |
|---|---|---|---|
| 1. SME (concise, fixed fix) | 4.93 | 5.00 | 1780 |
| 2. DE expert (with example) | 4.87 | 5.00 | 2177 |
| 3. Error-resolution (terse) | 4.60 | 4.80 | 1605 |
| 4. Structured (current default) | **4.97** | 5.00 | 1766 |

**Winner: Structured (current default).** It scored highest on groundedness, tied for the top
relevance score, and did so at a similar token cost to the SME persona (i.e. not by simply writing
more words). No prompt change is needed — the app's live default is already the best-scoring
option out of the 4 compared. The error-resolution persona (solution-only, no explanation) scored
noticeably lower on both dimensions — being terse apparently makes it easier to omit a caveat or
state a fix slightly too confidently, which the judge penalizes.

**Being upfront about how strong this result is:** the margins between personas 1, 2, and 4 are
small (within 0.1 points of 5), and 30 questions is not a large sample — a handful of answers
flipping their score would narrow or reverse the ranking among the top 3. The same kind of AI model
both wrote and judged the answers, which can introduce a bias toward whichever style that
particular AI model happens to favor, and structured, clearly-labeled output is a known bias for
AI judges specifically. The clearer, larger-margin gap is persona 3 (error-resolution) scoring
visibly lower than the other three — that part of the result is on firmer ground than the exact
ranking of 1/2/4. The search-method conclusion in step 2 is on much firmer ground than anything
from this step.

```powershell
uv run python evaluation/3_llm_prompt_judge_evaluation.py
uv run python evaluation/3_llm_prompt_judge_evaluation.py --n 10     # smaller, faster test run
```

## Files behind these results

None of these files are needed to just use the app — they're the underlying evidence behind the
numbers above.

| File | What it holds |
|---|---|
| `evaluation/test_questions_for_eval.json` | The 103 (question, correct write-up) pairs used in steps 1, 2, and 3 — 80 AI-generated once and reused as a fixed set, plus 23 merged in from `test_questions.md`'s Spark config-tuning questions |
| `evaluation/results_1_grid_search_evaluation.json` | Full comparison of all 1,024 search-weight combinations tried in step 1, with the winner at the top under `"best_result"` |
| `evaluation/results_2_retrieval_evaluation.json` | Full search-accuracy results from step 2, both parts A and B |
| `evaluation/results_3_llm_prompt_judge_evaluation.json` | Every scored answer from the step 3 prompt comparison |

## How this compares to the course, and how much to trust these results

The LLM Zoomcamp course's evaluation module teaches generating test questions from the documents
themselves (so the "correct answer" is known for free), then measuring how often each search style
finds it, and separately tuning search weights via a systematic grid search. This project follows
both practices — but a few honest differences and caveats are worth calling out:

- **Fewer questions per write-up than the course's own approach.** The course generates 5 questions
  per document; this project used 1 AI-generated question per sampled write-up plus a smaller set
  of human-written ones, 103 total across the 160-write-up corpus. A single question flipping from
  right to wrong shifts the headline numbers by a couple of percentage points — a real limit on how
  much weight these specific numbers can bear.
- **The step 3 judge scores against the retrieved write-ups, not a separate "correct answer."** The
  course's own AI-grading lesson compares a generated answer to the source's own stated answer. This
  project's judge instead checks whether an answer is *backed up by* the write-ups it was given —
  closer to how a live system would actually be checked, but it means "backed up by sources" and
  "objectively the right fix" are being conflated; only a human domain expert could catch the
  difference, and none reviewed these results.
- **The course warns that near-100% scores should raise suspicion, not confidence.** An earlier,
  AI-question-only version of step 2 showed exactly that warning sign (meaning-based search scoring
  a near-perfect Hit Rate) — which is exactly why the 45 hand-written questions in part B matter:
  they're the harder, more realistic check that caught it, and the final numbers above reflect that
  corrected picture.
- **The AI judge in step 3 was never checked against a human's own judgment.** There's no measure
  of how often its scores agreed with what a person would say looking at the same answers — the
  cheapest way to close this gap would be having someone read 15-20 of the judge's actual verdicts
  (favoring the lowest-scored ones) and reporting how often they agree with its reasoning.

**Bottom line:** the combined-search conclusion (step 2) is well supported — two independent
question sets agree, by a clear margin, with a sensible explanation for why. The
structured-answer-format conclusion (step 3) is weaker: it wins outright across all 4 personas, but
the margin over the next two personas is thin enough that a different sample of questions could
plausibly reorder them. Treat the search-method result as solid, and the exact 1st/2nd/3rd persona
ranking as a reasonable tie-breaker rather than a strong, proven result — the clearer, more durable
finding from step 3 is that the terse error-resolution persona underperforms the other three.

## Where this testing is still weak
- **The Monitoring page's activity mostly reflects these test runs**
- **Half of the write-ups were written with AI assistance rather than being real incident
  reports.** These were checked for having the right structure, but not individually fact-checked
  by a human expert.
