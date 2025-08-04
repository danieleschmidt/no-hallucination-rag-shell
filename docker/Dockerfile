# Multi-stage Docker build for No-Hallucination RAG Shell

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY pyproject.toml /tmp/
WORKDIR /tmp
RUN pip install --upgrade pip setuptools wheel && \
    pip install . && \
    pip install transformers sentence-transformers faiss-cpu \
        numpy pandas pydantic click rich aiohttp \
        asyncio-throttle python-dotenv pyyaml requests \
        beautifulsoup4 nltk scikit-learn torch safetensors

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r rag && useradd -r -g rag rag

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create directories
RUN mkdir -p /opt/no-hallucination-rag/{data,cache,models} \
    /var/log/no-hallucination-rag \
    && chown -R rag:rag /opt/no-hallucination-rag /var/log/no-hallucination-rag

# Copy application code
COPY --chown=rag:rag . /opt/no-hallucination-rag/
WORKDIR /opt/no-hallucination-rag

# Install the package
RUN pip install -e .

# Switch to non-root user
USER rag

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from no_hallucination_rag import FactualRAG; rag=FactualRAG(); print('OK')" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "no_hallucination_rag.shell.cli"]