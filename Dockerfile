FROM python:3.12-slim

# HF Spaces builds run as a non-root user by convention; this avoids
# permission errors on the mounted filesystem at runtime.
RUN useradd -m appuser
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces' Docker SDK expects the app to listen on 7860.
EXPOSE 7860

CMD ["uv", "run", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]