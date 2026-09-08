FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Build context must be the repo root:
#   docker build -f baseline/Dockerfile -t baseline .
COPY pyproject.toml uv.lock* ./
COPY scoring ./scoring
COPY baseline ./baseline

RUN uv sync --frozen --package breathing-baseline --no-dev

ENTRYPOINT ["uv", "run", "python", "-m", "breathing_baseline.predict"]
