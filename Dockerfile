# Phase 1: Echo / CPU image (§6.1)
# Base pinned by tag (pin by digest for production)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /srv

# Install uv and dependencies first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && \
    uv venv && \
    uv pip install --no-cache-dir -r pyproject.toml

# Copy application code
COPY app/ app/

# Create non-root user
RUN useradd -m app && chown -R app /srv
USER app

EXPOSE 8000

# Default entrypoint starts the server (§6.1: PS requires default entrypoint)
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "0"]
