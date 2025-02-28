#!/bin/bash
set -e

# Display GPU information
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "GPU Information:"
    nvidia-smi
else
    echo "Warning: nvidia-smi not found. Running without GPU support."
fi

# Check CUDA availability through Python
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); \
    if torch.cuda.is_available(): print(f'CUDA Device: {torch.cuda.get_device_name(0)}')"

# Check if model downloads are needed and download them
if [ "${DOWNLOAD_MODELS_ON_START:-false}" = "true" ]; then
    echo "Checking for model downloads..."
    python3 -c "
import json
import os
import requests
from pathlib import Path

# Load config
config_file = os.environ.get('CONFIG_FILE', 'model_config.json')
with open(config_file, 'r') as f:
    config = json.load(f)

# Check and download models
for model_type, model_config in config.items():
    if model_config.get('download_if_missing', False) and model_config.get('model_type') == 'gguf':
        model_path = model_config.get('model_path')
        model_url = model_config.get('model_url')
        
        if model_path and model_url and not os.path.exists(model_path):
            print(f'Downloading {model_type} model from {model_url}')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Stream download with progress reporting
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                if downloaded % (10 * 1024 * 1024) == 0:  # Report every 10MB
                                    print(f'Download progress: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB/{total_size/(1024*1024):.1f}MB)')
            
            print(f'Download completed: {model_path}')
    
    # Check embedding models
    if model_config.get('download_if_missing', False) and model_config.get('model_type') == 'huggingface':
        model_path = model_config.get('model_path')
        if model_path and not os.path.exists(model_path):
            print(f'Model path {model_path} will be downloaded on first use')
"
fi

# Apply ulimits for optimal performance
ulimit -n 65536
ulimit -l unlimited

# Set number of workers based on CPU cores if not specified
if [ -z "${WORKERS}" ]; then
    WORKERS=$(( $(nproc) / 2 ))
    if [ "$WORKERS" -lt 1 ]; then
        WORKERS=1
    fi
    echo "Auto-configured workers: $WORKERS"
fi

# Configure memory management for A100
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Execute the command
exec "$@"
