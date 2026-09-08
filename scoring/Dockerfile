FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies (cached layer)
COPY pyproject.toml .
RUN uv pip install --system -e ".[dev]"

# Copy source
COPY . .

ENTRYPOINT ["python", "/app/score.py"]
