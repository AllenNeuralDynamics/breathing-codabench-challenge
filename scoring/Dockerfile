FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ARG VERSION=42.42.42
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION

COPY . .
RUN uv sync --package scoring --no-dev --frozen --no-cache

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "/app/scoring/score.py"]
CMD ["/input", "/output"]
