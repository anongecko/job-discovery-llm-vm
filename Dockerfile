FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model_server/ /app/model_server/
COPY model_config.json /app/model_config.json

# Create directories
RUN mkdir -p /app/models

# Set environment variables
ENV CONFIG_FILE=/app/model_config.json

# Expose port
EXPOSE 8080

# Set entrypoint
CMD ["python", "-m", "model_server.main"]
