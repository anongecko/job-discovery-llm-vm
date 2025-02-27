# AI Model Server Setup Guide

This guide explains how to set up the AI Model Server for the Job Discovery System on a single Azure VM.

## Prerequisites

- Azure VM with GPU (recommended: Standard_NC6s_v3 or Standard_NC4as_T4_v3)
- Ubuntu 20.04 LTS or higher
- At least 16GB RAM (32GB+ recommended)
- 100GB+ SSD storage
- NVIDIA GPU drivers installed

## Installation Options

There are three ways to install and run the model server:

1. **Docker (Recommended)**: Easiest option with all dependencies packaged
2. **Direct Installation**: Manual installation using the setup script
3. **Development Setup**: For local development and testing

## Option 1: Docker Installation

1. Install Docker and NVIDIA Container Toolkit:

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
```

2. Clone or copy the model server files:

```bash
# Create directory
mkdir -p /opt/model-server
cd /opt/model-server

# Copy files (replace with your method of transferring files)
# ...

# Create models directory
mkdir -p models
```

3. Run using Docker Compose:

```bash
# Start the server
docker-compose up -d

# Check logs
docker-compose logs -f
```

4. Get the generated API key:

```bash
docker-compose exec model-server cat /app/api_keys.json
```

## Option 2: Direct Installation

1. Install system dependencies:

```bash
# Update package list
sudo apt-get update

# Install Python and other dependencies
sudo apt-get install -y python3 python3-pip python3-venv git wget
```

2. Clone or copy the model server files:

```bash
# Create directory
sudo mkdir -p /opt/model-server
cd /opt/model-server

# Copy files (replace with your method of transferring files)
# ...
```

3. Run the setup script:

```bash
# Make the script executable
chmod +x setup.sh

# Run the setup script
sudo ./setup.sh
```

4. The API key will be saved to `/opt/model-server/admin_key.txt`.

5. Manage the service:

```bash
# Check status
sudo systemctl status model-server

# Start/stop/restart
sudo systemctl start model-server
sudo systemctl stop model-server
sudo systemctl restart model-server
```

## Option 3: Development Setup

1. Install system dependencies:

```bash
# Update package list
sudo apt-get update

# Install Python and other dependencies
sudo apt-get install -y python3 python3-pip python3-venv git wget
```

2. Set up virtual environment:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. Run the server:

```bash
# Run with uvicorn
python -m model_server.main
```

## Testing the Installation

1. Use the test client to verify the installation:

```bash
# Install dependencies for test client
pip install httpx

# Run tests (replace API_KEY with your key)
python test_client.py --api-key API_KEY --all
```

2. Check the API documentation:
   - Open `http://<your-vm-ip>:8080/docs` in a web browser

## Configuring the Job Discovery System

1. Update your Job Discovery System `.env` file with:

```
PRIMARY_MODEL_ENDPOINT=http://<your-vm-ip>:8080
PRIMARY_MODEL_KEY=<your-api-key>
PRIMARY_MODEL_DEPLOYMENT=v1/completions
PRIMARY_MODEL_VERSION=""  # Not needed for this server

EMBEDDING_MODEL_ENDPOINT=http://<your-vm-ip>:8080
EMBEDDING_MODEL_KEY=<your-api-key>
EMBEDDING_MODEL_DEPLOYMENT=v1/embeddings
EMBEDDING_MODEL_VERSION=""

CLASSIFIER_MODEL_ENDPOINT=http://<your-vm-ip>:8080
CLASSIFIER_MODEL_KEY=<your-api-key>
CLASSIFIER_MODEL_DEPLOYMENT=v1/classifications
CLASSIFIER_MODEL_VERSION=""
```

## Model Configuration

The server uses `model_config.json` to configure which models to load. The default configuration uses:

- **Primary Model**: Mixtral-8x7B-Instruct-v0.1 (4-bit quantized)
- **Embedding Model**: all-MiniLM-L6-v2
- **Classifier Model**: gemma-2b-it (4-bit quantized)

To modify the models:

1. Edit `model_config.json`
2. Restart the server:
   - Docker: `docker-compose restart`
   - Systemd: `sudo systemctl restart model-server`

## Troubleshooting

### Common Issues

1. **Out of CUDA memory**: The VM doesn't have enough GPU memory
   - Adjust quantization in `model_config.json` (use 8-bit instead of 4-bit)
   - Use a VM with more GPU memory
   
2. **API key errors**: Check the API key is correct
   - Verify the key matches one in `api_keys.json`
   - Create a new key if needed: 
     ```python
     from model_server.auth import generate_new_key
     print(generate_new_key("name", "role"))
     ```

3. **Models downloading slowly**: Models are fetched from Hugging Face the first time
   - Be patient on first run
   - Use a VM with good internet connectivity
   - Consider downloading models manually and placing them in the `models` directory

### Logs

Check the logs to diagnose problems:

- Docker: `docker-compose logs -f model-server`
- Systemd: `sudo journalctl -u model-server -f`

## Performance Optimization

1. **Memory Usage**:
   - Use 4-bit quantization for large models
   - Run the embedding model on CPU if needed
   
2. **Speed**:
   - Increase `max_batch_size` for embedding model
   - Consider using smaller models for the classifier

## Security Notes

1. Secure the API key properly
2. Consider using HTTPS if exposing the server outside your network
3. Implement proper firewall rules on the VM

## Scaling Considerations

If you need to handle more requests:

1. Increase `max_batch_size` in model config
2. Use a VM with more CPU cores 
3. Consider separating models to different VMs if needed

## Resource Monitoring

Monitor resource usage with:

```bash
# GPU utilization
nvidia-smi -l 5

# System resources
htop
```

The server also provides metrics at `/metrics` endpoint (requires API key).
