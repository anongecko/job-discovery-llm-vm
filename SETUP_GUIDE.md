# Enhanced AI Model Server Setup Guide

This guide explains how to set up the AI Model Server for the Job Discovery System on an Azure NC24ads A100 v4 VM.

## System Requirements

- **Azure VM**: NC24ads A100 v4 (or equivalent with A100 GPU)
- **GPU**: NVIDIA A100 (80GB VRAM)
- **CPU**: 24 vCPUs
- **System Memory**: 220 GiB
- **Storage**: At least 200GB SSD (premium disk recommended)
- **OS**: Ubuntu 20.04 LTS or higher
- **CUDA**: 11.8 or higher
- **Docker**: Latest with NVIDIA Container Toolkit

## Pre-Installation Checklist

Before starting the installation, ensure that:

1. Your VM has the NVIDIA drivers installed
2. CUDA 11.8 or higher is installed
3. Docker and NVIDIA Container Toolkit are installed (for Docker deployment)
4. You have access to pull from Hugging Face (for model downloads)

If your VM doesn't have these prerequisites, follow these steps:

### Install NVIDIA Drivers and CUDA

```bash
# Install NVIDIA drivers
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot

# After reboot, install CUDA
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit
```

### Install Docker and NVIDIA Container Toolkit

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Configure Docker for NVIDIA runtime
sudo tee /etc/docker/daemon.json > /dev/null << EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

## Installation Methods

There are two ways to install and run the model server:

1. **Docker Installation (Recommended)**: Best for production deployments
2. **Direct Installation**: For development or custom configurations

## Docker Installation (Recommended)

1. Clone or copy the model server files:

```bash
# Create directory
mkdir -p /opt/model-server
cd /opt/model-server

# Copy files (replace with your method of transferring files)
# ...
```

2. Run the automated setup script:

```bash
chmod +x setup.sh
sudo ./setup.sh
```

3. When prompted, choose whether to pre-download models:
   - Select **Yes** if you want to download models during setup
   - Select **No** to download models on first use

4. Verify the installation:

```bash
# Check service status
sudo systemctl status model-server-docker.service

# Check container logs
sudo docker-compose -f /opt/model-server/docker-compose.yml logs -f
```

## Direct Installation (Advanced)

1. Install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git wget build-essential cmake libopenblas-dev libomp-dev
```

2. Copy files to installation directory:

```bash
sudo mkdir -p /opt/model-server
sudo cp -r model_server /opt/model-server/
sudo cp model_config.json requirements.txt /opt/model-server/
```

3. Set up virtual environment:

```bash
sudo python3 -m venv /opt/model-server/venv
sudo /opt/model-server/venv/bin/pip install --upgrade pip
sudo /opt/model-server/venv/bin/pip install -r /opt/model-server/requirements.txt
```

4. Create service user and set permissions:

```bash
sudo useradd -r -s /bin/false ai-service
sudo chown -R ai-service:ai-service /opt/model-server
sudo chmod -R 750 /opt/model-server
```

5. Create systemd service:

```bash
sudo tee /etc/systemd/system/model-server.service > /dev/null << EOF
[Unit]
Description=AI Model Server for Job Discovery
After=network.target

[Service]
User=ai-service
Group=ai-service
WorkingDirectory=/opt/model-server
ExecStart=/opt/model-server/venv/bin/python -m model_server.main
Restart=on-failure
RestartSec=5
Environment=PYTHONPATH=${PYTHONPATH}:/opt/model-server
Environment=CONFIG_FILE=/opt/model-server/model_config.json
Environment=WORKERS=8

# Increase limits for A100 GPU
LimitMEMLOCK=infinity
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable model-server.service
sudo systemctl start model-server.service
```

## Model Information

The server uses three AI models optimized for the A100 GPU:

### 1. Primary Extraction Model
- **Model**: Llama 3.1 70B Instruct (Q6_K)
- **Type**: GGUF Quantized
- **Size**: ~45GB on disk
- **VRAM Usage**: ~25-30GB
- **Source**: [TheBloke/Llama-3.1-70B-Instruct-GGUF](https://huggingface.co/TheBloke/Llama-3.1-70B-Instruct-GGUF)

### 2. Embedding Model
- **Model**: Nomic-Embed-Text-v1
- **Type**: HuggingFace
- **VRAM Usage**: ~2GB
- **Source**: [nomic-ai/nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1)

### 3. Fast Classifier Model
- **Model**: Mistral 7B Instruct v0.2 (Q4_K_M)
- **Type**: GGUF Quantized
- **Size**: ~4.5GB on disk
- **VRAM Usage**: ~4GB
- **Source**: [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

## API Usage

The model server provides three main endpoints:

### 1. Text Generation
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "prompt": "Extract the job title, company name, and location from this job listing: \"Senior Software Engineer at Google in Mountain View, CA\"",
    "system_message": "You are an expert job data extractor.",
    "max_tokens": 1024,
    "temperature": 0.1,
    "stream": false
  }'
```

### 2. Embeddings Generation
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "input": ["Software Engineer with experience in Python and JavaScript", "Data Scientist with experience in machine learning"]
  }'
```

### 3. Text Classification
```bash
curl -X POST http://localhost:8080/v1/classifications \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "text": "Looking for a software engineer with 5 years of experience in Python and JavaScript",
    "categories": ["job_posting", "resume", "cover_letter", "job_application"]
  }'
```

## Monitoring

The server includes Prometheus and Grafana for monitoring:

1. **Prometheus**: Access at http://your-vm-ip:9090
2. **Grafana**: Access at http://your-vm-ip:3000 (default credentials: admin/model-server-admin)

Grafana is pre-configured with dashboards for:
- Model server metrics
- GPU utilization
- API latency and throughput
- Memory usage

## Job Discovery System Integration

To connect the Job Discovery System to this model server:

1. Update your Job Discovery System `.env` file:

```
MODEL_SERVER_ENDPOINT=http://<your-vm-ip>:8080
MODEL_SERVER_API_KEY=<your-api-key>
```

2. Update your configuration to use the appropriate model endpoints:

```
PRIMARY_MODEL_ENDPOINT=${MODEL_SERVER_ENDPOINT}/v1/completions
EMBEDDING_MODEL_ENDPOINT=${MODEL_SERVER_ENDPOINT}/v1/embeddings
CLASSIFIER_MODEL_ENDPOINT=${MODEL_SERVER_ENDPOINT}/v1/classifications
```

## Performance Optimization

For optimal performance on the A100 GPU:

1. **Memory Management**:
   - The A100 with 80GB VRAM can handle all models simultaneously
   - Primary model is configured to use ~30GB VRAM
   - Embedding and classifier models share the remaining VRAM

2. **Concurrent Requests**:
   - Primary model: Up to 12 concurrent requests
   - Embedding model: Up to 24 concurrent requests
   - Classifier model: Up to 16 concurrent requests

3. **Worker Configuration**:
   - For 24 vCPUs, 8 worker processes are optimal
   - Each worker can handle multiple concurrent requests

4. **Request Batching**:
   - Embeddings support batches of up to 64 texts
   - Use batching for embedding large datasets

## Troubleshooting

### Common Issues

1. **Out of memory errors**:
   - Check GPU memory usage: `nvidia-smi`
   - Adjust `n_gpu_layers` in model_config.json if needed
   - Reduce `max_concurrent_requests` settings

2. **Slow initialization**:
   - First request after startup loads models into memory
   - This can take 1-2 minutes for the primary model
   - Subsequent requests will be much faster

3. **API key errors**:
   - Verify the key matches one in `/opt/model-server/api_keys.json`
   - Create a new key with:
     ```
     cd /opt/model-server
     sudo -u ai-service python -c "from model_server.auth import generate_new_key; print(generate_new_key('client', 'client'))"
     ```

### Logs

Check logs for detailed error information:

```bash
# For Docker deployment
sudo docker-compose -f /opt/model-server/docker-compose.yml logs -f

# For direct installation
sudo journalctl -u model-server -f
```

## Security Considerations

To secure your model server in production:

1. **Network Security**:
   - Use a firewall to restrict access to port 8080
   - Consider setting up a reverse proxy with TLS termination

2. **API Key Management**:
   - Regenerate the default admin key after setup
   - Create separate keys for each client application
   - Revoke unused keys

3. **Container Security**:
   - Keep Docker and the model server updated
   - Run regular security scans

## Maintenance

### Updates

To update the model server:

1. Stop the service:
   ```bash
   sudo systemctl stop model-server-docker.service
   ```

2. Update files:
   ```bash
   cd /opt/model-server
   # Update files as needed
   ```

3. Restart the service:
   ```bash
   sudo systemctl start model-server-docker.service
   ```

### Model Updates

To update models:

1. Edit `model_config.json` with new model paths and URLs
2. Delete old model files if needed
3. Restart the service to download new models

## Advanced Configuration

For advanced configurations, modify the following files:

1. `model_config.json`: Model parameters and paths
2. `docker-compose.yml`: Container settings
3. `prometheus.yml`: Monitoring configuration

## Support

For issues, check:
1. Server logs first
2. GPU status and memory usage
3. System resources (CPU, RAM, disk space)

## Resource Monitoring

Monitor resource usage with:

```bash
# GPU stats
nvidia-smi -l 5

# Container stats
docker stats
```

The server also provides metrics at `/metrics` endpoint (requires API key).
