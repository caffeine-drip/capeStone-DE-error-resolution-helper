# Docker Deployment Guide

This guide walks through running the DE IncidentIQ using Docker.

**What is Docker, in plain terms?** Docker is a tool that packages an entire app — the code, the
programming language it needs, and every dependency — into one bundle called a "container," so it
runs the same way on any computer without you having to install anything by hand. Instead of
installing Python and a dozen libraries yourself, you install Docker once, and it does the rest.

**What is a terminal / command line?** It's a text-based way of giving your computer instructions,
instead of clicking buttons. Every gray box below is a command: you type or paste the text into
the terminal window and press Enter, and the computer runs it. This guide explains what each one
does before you run it.

This guide covers all three ways to answer questions (a free model on your own computer, or two
paid cloud services), explains what happens at each step, and has a troubleshooting section at the
end for common problems.

## What runs inside the container vs. what stays on your computer

- **Inside the container:** the app itself and all of its code — everything needed to answer a
  question.
- **On your own computer, but reached over the network:** if you choose to run the AI model
  yourself for free (instead of paying for a cloud service), that model runs directly on your
  computer, not inside the container. Docker just knows how to reach it.
- **Kept on your computer, not sealed inside the container:** the knowledge base and the record of
  past questions and answers. This means your data survives even if you stop or rebuild the
  container.

## 1. What you'll need before starting

- **Docker Desktop** if you're on Windows or Mac, or **Docker Engine** if you're on Linux (this is
  the program that runs containers — install it first from Docker's website if you don't have it).
- A copy of this project's folder on your computer, with its `data` folder intact — the knowledge
  base files are already included, so you don't need to build anything before running the app.
- **One of the following**, depending on how you want to get answers:
  - a **DeepSeek account and API key** (a per-use paid cloud AI service — cheapest way to get a
    fully working system), or
  - an **OpenAI account and API key** (another paid cloud AI service), or
  - **a capable graphics card on your own computer**, to run the AI model yourself for free (see
    step 5).

An "API key" is just a private password-like code a cloud AI service gives you so it knows to bill
your account when the app asks it a question.

## 2. Get the project

```powershell
git clone https://github.com/caffeine-drip/capeStone-DE-error-resolution-helper.git
cd capeStone-DE-error-resolution-helper
```

## 3. Choose which AI service to use

The project includes a template settings file. Copy it to create your real settings file:

```powershell
copy .env.example .env
```

This command copies a file named `.env.example` and creates a new file named `.env` from it — the
file where you'll write your settings.

Open `.env` in any text editor and set this line to whichever service you plan to use:

```
LLM_PROVIDER=deepseek      # or: openai | local
```

**When running in Docker, this defaults to `deepseek` if you don't set it at all.** That's because
a Docker container has no graphics card of its own to run a model locally, so it makes sense to
default to a cloud service. Only set this to `local` if you're going to run an AI model on your own
computer and point the container at it (step 5).

This setting only decides which option is pre-selected when the app opens — you can still switch
between services from inside the running app afterward (step 7), as long as you've set up the
matching key first.

## 4. Set your real API keys — never write them into any project file

Your actual API key values should never be typed into `.env` or any file in this project. Instead,
you set them as "environment variables" — private settings that live on your computer, outside any
file that could accidentally be shared or backed up with the project.

**Windows (PowerShell), just for your current terminal window:**
```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
$env:OPENAI_API_KEY   = "sk-..."
```
Replace `sk-...` with your actual key. This only lasts until you close the terminal window.

**Windows, saved permanently — then open a brand-new terminal window before continuing:**
```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sk-...", "User")
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY",   "sk-...", "User")
```
This saves the key so you don't have to set it again next time, but it only takes effect in
terminal windows you open *after* running this.

**Mac/Linux (add the line to your shell's startup file, then reopen your terminal):**
```bash
export DEEPSEEK_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
```

If you're only planning to run the AI model locally on your own computer (step 5), you can skip
this step entirely — the app works fine without either key, it just won't offer the paid options
as working choices.

## 5. If running the AI model yourself: start it before starting the app

Only needed if you set `LLM_PROVIDER=local`. Skip this whole step if you're using DeepSeek or
OpenAI.

This starts two small local servers on your own computer: one that writes the answers, and one
that helps the search feature understand the *meaning* of your question (not just matching exact
words). Each command below downloads/loads a model file and starts a background process listening
on a specific "port" (a numbered channel Docker will know how to reach).

```powershell
# Answer-writing model
llama-server.exe -m Qwen3.5-9B-Q4_K_M.gguf `
  --port 11434 --ctx-size 32768 -ngl 999 --flash-attn on `
  --parallel 1 --batch-size 2048 --jinja --reasoning-format none

# Meaning-search model (used for smarter search, regardless of which AI writes the answer)
llama-server.exe -m Qwen3-Embedding-0.6B-Q8_0.gguf `
  --port 11435 --embedding -ngl 999 --ctx-size 8192
```

Please change the flags above based on your own hardware — `-ngl` (how many layers get offloaded
to your graphics card), `--ctx-size`, `--batch-size`, and `--parallel` all depend on how much GPU
memory and compute you actually have available, and the values shown are just what this project
happened to run with. Swap in whichever quantized model file fits your setup.

You don't need to change any settings for the container to find these — that's already configured.
Just make sure both commands above are running before you start the app in step 6.

**Note:** the second server (meaning-search) is useful no matter which AI service answers your
question. If it isn't running, the app automatically falls back to plain exact-word search instead
— nothing breaks, search results are just a little less smart without it.

## 6. Build and start the app

From the project's main folder (the one containing `docker-compose.yml`), run:

```powershell
docker compose up --build
```

This tells Docker to assemble the container (installing everything the app needs) and then start
it. The first time you run this, it takes a few minutes. Every time after that is much faster,
since Docker remembers what it already installed.

Once it's running, open your web browser and go to **http://localhost:8501** to use the app.

To run it in the background instead, so your terminal window is free to use for something else:
```powershell
docker compose up --build -d
```

## 7. Using the app

- Near the top of the sidebar, a switch lets you choose which AI service answers your questions.
  It starts on whichever one you set in step 3, but you can change it any time the app is open —
  no restart needed.
- If you pick a service whose key you never set up in step 4, the app shows a warning message
  instead of crashing.
- Everything else in the app (filtering results, adjusting how many write-ups it searches through,
  turning the self-check step on or off, and the Monitoring page) works exactly the same as running
  it outside Docker.

## 8. Your data is kept safe on your own computer

The app's knowledge base and its record of past questions live in a folder on your computer, not
sealed inside the container. In practice this means:

- The log of past questions, answers, and your thumbs-up/down feedback survives even if you stop
  and restart the container.
- The searchable knowledge base is built automatically the first time you ask a question, from the
  write-ups already included in the project.
- If you add or update write-ups later, the app notices the change automatically the next time you
  ask a question — you don't need to manually tell it to reload anything.

## 9. Stopping, restarting, and rebuilding

```powershell
docker compose down                 # stop the app (your data is untouched)
docker compose up -d                # start it again (fast — nothing needs rebuilding)
docker compose up --build -d        # rebuild after you've changed the app's code or dependencies
```

**If you change `.env`** (for example, switching which AI service is selected by default), you need
to recreate the container for the change to take effect — simply restarting isn't enough:

```powershell
docker compose up -d --force-recreate
```

(Or just switch services from inside the running app instead, as described in step 7 — that works
immediately with no restart at all.)

## 10. Troubleshooting

**"DEEPSEEK_API_KEY is not set" even though I set it.**
You probably set it in a different terminal window than the one you used to run
`docker compose up`, or you saved it permanently (the second option in step 4) without opening a
brand-new terminal window afterward. Type `echo $env:DEEPSEEK_API_KEY` (Windows) or
`echo $DEEPSEEK_API_KEY` (Mac/Linux) in the *same* window you're about to run Docker from to check
it's actually set there, then recreate the container (step 9).

**Using the local option, but getting a "connection refused" error.**
- Double check the two model servers from step 5 are actually running.
- If you changed which ports those servers use, you'll need to update the matching settings in
  `.env` (see the examples inside `.env.example`).

**The app starts but says it can't find the write-ups / knowledge base.**
Make sure the project's `data` folder — with its knowledge base files — is present in your project
folder before running `docker compose up`. These files are read from your computer, not built into
the container image itself.

**The container builds, but the app immediately stops or shows an error.**
Check what happened with `docker compose logs app` — this prints out the app's own log messages,
which usually point to the exact problem (often a missing file in `data`, or a setup issue). If
you're not sure what a message means, you can also try `docker compose build --no-cache` to rule
out a stale, half-built version of the container.

**I changed the app's code and nothing happened.**
You need to run `docker compose up --build` (or `--build -d`), not just `up` — plain `up` reuses
whatever container was already built, so it won't pick up code changes on its own.
