# Base image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install essential build tools
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /

# Copy dependency files first (for better caching)
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY LICENSE LICENSE

# Install dependencies with cache mount for faster rebuilds
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-cache --no-install-project

# Copy application code
COPY src/ src/
COPY models/ models/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    HOST=127.0.0.1 \
    PORT=8000
# Expose the API port
EXPOSE 8000

# Run the FastAPI application with uvicorn
ENTRYPOINT ["uv", "run", "uvicorn", "ml_ops_59.api:app", "--host", "127.0.0.1", "--port", "8000"]
