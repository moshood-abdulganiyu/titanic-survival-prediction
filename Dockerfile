FROM python:3.12-slim

# Running as non-root is good practice for any container host, not just
# HF Spaces: it limits blast radius if the app is ever compromised.
RUN useradd -m appuser
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .
RUN chown -R appuser:appuser /app
USER appuser

# Render assigns the listen port at runtime via $PORT, it isn't fixed like
# HF Spaces' 7860. EXPOSE is documentation only (doesn't bind anything),
# 8000 here just matches the default used for local `docker run` testing.
EXPOSE 8000

# Shell form (not exec/array form) so $PORT actually expands. Falls back to
# 8000 if PORT isn't set, e.g. running this image locally with `docker run`.
CMD uv run uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}