FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install build tools (often needed for scientific deps)
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock /app/

# Install deps (locked)
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Copy application code + artifacts
COPY src/ /app/src/
COPY models/ /app/models/
# If you also need configs/data files at runtime, add them too:
# COPY configs/ /app/configs/

# Cloud Run provides PORT env var; must bind to it
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080

# IMPORTANT: correct module path: src.ml_ops_59.api:app
CMD ["uv", "run", "uvicorn", "src.ml_ops_59.api:app", "--host", "0.0.0.0", "--port", "8080"]




# # Base image with uv pre-installed
# FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# # Install essential build tools
# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

# # Set working directory
# WORKDIR /

# # Copy dependency files first (for better caching)
# COPY uv.lock uv.lock
# COPY pyproject.toml pyproject.toml
# COPY README.md README.md
# COPY LICENSE LICENSE

# # Install dependencies with cache mount for faster rebuilds
# ENV UV_LINK_MODE=copy
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --locked --no-cache --no-install-project

# # Copy application code
# COPY src/ src/
# COPY models/ models/

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     HOST=0.0.0.0 \
#     PORT=8000

# # Expose the API port
# EXPOSE 8000

# # Run the FastAPI application with uvicorn
# ENTRYPOINT ["uv", "run", "uvicorn", "ml_ops_59.api:app", "--host", "0.0.0.0", "--port", "8000"]
