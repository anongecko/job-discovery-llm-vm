FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies with cleanup to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    curl \
    wget \
    git \
    libopenblas-dev \
    libomp-dev \
    pkg-config \
    libcairo2-dev \
    libxml2-dev \
    libpango1.0-dev \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip and install python dependencies with version pinning
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model_server/ /app/model_server/
COPY model_config.json /app/model_config.json

# Create directories and set permissions
RUN mkdir -p /app/models && \
    chmod -R 755 /app

# Set environment variables for model server
ENV CONFIG_FILE=/app/model_config.json \
    PYTHONPATH=/app \
    WORKERS=8 \
    PORT=8080 \
    HOST=0.0.0.0

# Set entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8080

# Set health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python", "-m", "model_server.main"]
