FROM python:3.12-slim

# uv gives fast, reproducible installs from the committed uv.lock (pinned
# versions) — same tool used for local dev, so container behavior matches.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (better layer caching — only re-runs when
# pyproject.toml/uv.lock change, not on every code edit).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# App code (app.py is the only file run directly; core pipeline code lives in
# my_assistant/) + the pre-built knowledge base (enriched_documents.json,
# embeddings.npy) — the DuckDB KB/trace stores get created on first run and
# should be volume-mounted (see docker-compose.yml) so they persist.
COPY app.py ./
COPY my_assistant/ ./my_assistant/
COPY data/enriched_documents.json data/embeddings.npy data/embeddings_ids.json ./data/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
