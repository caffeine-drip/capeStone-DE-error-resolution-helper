# Running Without Docker

This guide walks through running the DE IncidentIQ directly on your own computer,
without Docker. If you'd rather run it inside Docker instead (a way of packaging the whole app so
you don't need to install anything by hand), see [`DockerDeployment.md`](DockerDeployment.md)
instead.

**What is a terminal / command line?** It's a text-based way of giving your computer instructions
instead of clicking buttons. Every gray box below is a command: you type or paste the text into
your terminal window and press Enter. This guide explains what each command does before you run
it.

## 1. What you'll need before starting

- **Python** (the programming language the app is written in) and a tool called **uv**, which
  installs everything else the app needs automatically. Install uv by following the instructions
  at [docs.astral.sh/uv](https://docs.astral.sh/uv/).
- **One of the following**, depending on how you want to get answers:
  - a **DeepSeek API key** (a per-use paid cloud AI service — cheapest way to get a fully working
    system, no special hardware needed), or
  - an **OpenAI API key** (another paid cloud AI service), or
  - **a computer with a capable graphics card**, so you can run the AI model yourself for free.
- You only need the following if you plan to add brand-new issues to the knowledge base yourself
  (section 7) — not needed just to use the app with the write-ups already included: a locally
  running AI model, or a DeepSeek/OpenAI key, used to generate the extra detail for new entries.

An "API key" is a private password-like code a cloud AI service gives you so it knows to bill your
account when the app asks it a question.

## 2. Get the project and install its dependencies

```powershell
git clone https://github.com/caffeine-drip/capeStone-DE-error-resolution-helper.git
cd capeStone-DE-error-resolution-helper
uv sync
```

`uv sync` reads the project's dependency list (`pyproject.toml`/`uv.lock`) and installs the exact
versions of everything the app needs.

## 3. Choose which AI service to use

The project includes a template settings file. Copy it to create your own real settings file:

```powershell
copy .env.example .env
```

Open the new `.env` file in any text editor and set this line to whichever service you plan to
use:

```
LLM_PROVIDER=local      # or: deepseek | openai
```

Running this way (without Docker), the setting defaults to `local` if you leave it unset — meaning
it assumes you want to run the AI model yourself, since that's easy to do on your own computer.
(Docker defaults the opposite way, to a paid cloud service, since a container has no graphics card
of its own.)

This setting only decides which option is pre-selected when the app opens — you can still switch
services from inside the running app afterward, as long as you've set up the matching key first
(section 4).

## 4. Set your real API keys — never write them into any project file

Your actual API key values should never be typed into `.env` or any other file in this project.
Instead, you set them as "environment variables" — private settings that live on your computer,
separate from any project file.

**Windows, saved permanently:**
```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "<your key>", "User")
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY",   "<your key>", "User")
```

**Mac/Linux (add these lines to your shell's startup file, e.g. `~/.bashrc` or `~/.zshrc`):**
```bash
export DEEPSEEK_API_KEY="<your key>"
export OPENAI_API_KEY="<your key>"
```

**Close and reopen your terminal window (and restart the app if it's already running) after
setting these** — the change only takes effect in new terminal sessions.

Neither key is required to run the app. If a cloud service's key is missing, the app shows a
warning only if you try to select that service, and the local option still works fine.

## 5. If running the AI model yourself: start it first

Skip this whole step if you're only using DeepSeek or OpenAI.

This starts two small local servers on your own computer: one that writes the answers, and one
that helps the search feature understand the *meaning* of your question (not just matching exact
words). `--jinja` is a required setting on the first one — without it, the model can't follow the
formatting instructions correctly and ends up returning nothing useful.

```powershell
# 1. Answer-writing model
llama-server.exe -m Qwen3.5-9B-Q4_K_M.gguf `
  --port 11434 --ctx-size 32768 -ngl 999 --flash-attn on `
  --parallel 1 --batch-size 2048 --jinja --reasoning-format none

# 2. Meaning-search model
llama-server.exe -m Qwen3-Embedding-0.6B-Q8_0.gguf `
  --port 11435 --embedding -ngl 999 --ctx-size 8192
```

Please change the flags above based on your own hardware — `-ngl` (how many layers get offloaded
to your graphics card), `--ctx-size`, `--batch-size`, and `--parallel` all depend on your available
GPU memory and compute, and the values above are just what this project happened to run with. Swap
in whichever quantized model file actually fits your setup.

To double-check the first server is working correctly, run:

```powershell
uv run python data_pipeline/enrich_documents.py --diagnose
```

This sends a few different test requests and reports which ones the server handles correctly.

**Using only DeepSeek or OpenAI?** Skip both servers above entirely. The app automatically falls
back to plain exact-word search if no meaning-search server is running — nothing breaks, search
results are just a little less smart without it. Set `LLM_PROVIDER=deepseek` or
`LLM_PROVIDER=openai` in `.env`, or just pick the service from inside the app.

## 6. Run the app

```powershell
uv run streamlit run app.py
```

This opens the app in your web browser at **http://localhost:8501**. The write-ups themselves are
already included in the project (`data/enriched_documents.json` + the pre-computed
`data/embeddings.npy`), so there's nothing for you to run or download first. The first time you ask
a question, the app automatically builds a local `data/knowledge_base.duckdb` file from those
included write-ups — you'll see it appear in the `data` folder after your first question. That file
(along with `data/rag_traces.duckdb`, which logs your questions for the Monitoring page) is
intentionally not included in the project or tracked in git — it's just a local cache/log that
rebuilds itself, not something you need to be given.

### Command-line alternative (no browser window)

If you'd rather ask a single question from the terminal instead of opening the browser interface:

```powershell
uv run python my_assistant/rag.py "Spark executor OOM during shuffle"
uv run python my_assistant/rag.py --deepseek "why is my broadcast join timing out?"
uv run python my_assistant/rag.py --openai "why is my broadcast join timing out?"
uv run python my_assistant/rag.py --keyword-only --no-check "shuffle spill to disk"
```

Optional switches you can add: `--deepseek` (use DeepSeek for just this run), `--openai` (use
OpenAI/ChatGPT for just this run — with neither flag, it follows whatever `LLM_PROVIDER` is set to
in `.env`), `--use-embeddings` (force meaning-based search on), `--keyword-only` (force exact-word
search only, skipping meaning-based search), `--vector-only` (meaning-based search only, skipping
exact-word search), `--no-check` (skip the self-check/rewrite step).

## 7. (Optional) Adding new issues to the knowledge base

Only needed if you want to add new write-ups yourself — skip this if you just want to use the app
as-is.

New issues are added directly to `data/all_documents.json`, the plain master copy of the knowledge
base, in the same simple format as the existing entries: an ID, a title, a category, the problem,
and the resolution.

Once you've added new entries there, run these two steps to prepare them for use:

```powershell
# Adds AI-generated detail (a summary, tags, etc.) and a "meaning fingerprint" to any
# entry that doesn't have one yet. Safe to re-run any time — it skips entries that
# are already done, so adding a few new issues doesn't reprocess everything.
uv run python data_pipeline/enrich_documents.py
uv run python data_pipeline/enrich_documents.py --limit 5        # try it on just 5 first

# Loads the updated knowledge base into the app's lightweight database
uv run python my_assistant/kb_ingest.py
```

If you don't have an AI model running (locally or via a key) when you run the first command, new
entries still get added — they just won't have a meaning-fingerprint yet, so they'll only be found
by exact-word search until you run it again later with a model available. Nothing breaks or stops
partway through.

## 8. (Optional) Checking how well the system performs

The project includes a separate set of tests that measure search accuracy and answer quality,
independent of normal day-to-day use. These are documented on their own in
[`retrivalLLMEvalTest.md`](retrivalLLMEvalTest.md), including the exact commands to run them — not
repeated here to avoid the two documents drifting out of sync.

## 9. Troubleshooting

**An error mentions a missing module/library (e.g. "No module named 'minsearch'").**
You probably ran `streamlit run app.py` or `python ...` directly instead of starting the command
with `uv run`. `uv run` makes sure the app uses the exact set of installed dependencies from step
2; running a plain `python` or `streamlit` command instead uses whatever generic Python is set up
on your computer, which won't have this project's dependencies installed.

**Using the local option, but getting a "connection refused" error.**
The two model servers from step 5 aren't running, or aren't on the expected ports. Start them
before running the app.

**The sidebar says a cloud service's API key isn't set, even though you set it.**
The key isn't set as a real environment variable in the same terminal window you launched the app
from. On Windows, keys saved permanently (the "saved permanently" option in step 4) only take
effect in terminal windows opened *after* you saved them — close and reopen your terminal, then
try again.

**The app says it can't find the knowledge base / says there are 0 documents.**
Make sure the project's `data` folder still contains its knowledge base files — this should only
happen if they were deleted or you're working from an incomplete copy of the project. If missing,
restore them or rebuild them following section 7.

**The AI model returns nothing / the app seems to hang while adding new issues.**
Make sure `--jinja` was included when you started the answer-writing model server in step 5 —
without it, the model can't format its response correctly and ends up returning nothing usable.
Run `uv run python data_pipeline/enrich_documents.py --diagnose` to check exactly what's going
wrong.

**I want to switch which AI service answers my questions, without restarting anything.**
Just use the switch inside the running app — as long as the relevant key was already set up
before you started the app, switching services takes effect immediately with no restart needed.
