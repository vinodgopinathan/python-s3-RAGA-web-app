# 🚀 Multi-stage build for faster subsequent builds and better caching
FROM python:3.11-slim as system-deps

# Install system dependencies only (cached layer - rarely changes)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Separate stage for build tools (cached layer)
FROM system-deps as build-deps

# Install build dependencies (heavier layer, but cached)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Dependencies stage - install Python packages separately for better caching
FROM build-deps as python-deps

# Copy only requirements first (this layer is cached until requirements change)
COPY requirements-docker-cpu.txt ./requirements.txt

# Install Python dependencies with optimized pip settings
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-deps --no-cache-dir wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf ~/.cache/pip && \
    find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Production stage - minimal final image
FROM system-deps as production

# Add build arguments
ARG AWS_REGION
ENV AWS_REGION=${AWS_REGION}

# Install only runtime dependencies for tesseract and pdf2image
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy Python packages from deps stage
COPY --from=python-deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Copy application code
COPY src ./src
COPY chunking_api.py ./
COPY database_schema.sql ./

# Create uploads directory and set permissions
RUN mkdir -p /app/uploads && \
    chown -R nobody:nogroup /app && \
    chmod -R 755 /app

# Set Python path to include the src directory
ENV PYTHONPATH=/app/src:/app

# Expose port
EXPOSE 5000

# Health check endpoint (update path)
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application (fixed path)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "--preload", "--chdir", "/app/src", "app:app"]
