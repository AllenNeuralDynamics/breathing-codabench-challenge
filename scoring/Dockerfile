FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Build context is the repo root

COPY . .
RUN uv sync --package scoring --no-dev --frozen --no-cache

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "/app/scoring/score.py"]
CMD ["/input", "/output"]
