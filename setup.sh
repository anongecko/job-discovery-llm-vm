#!/bin/bash

# Enhanced Setup script for AI Model Server on A100 VM
set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit 1
fi

# Set working directory
INSTALL_DIR="/opt/model-server"
echo "Installing AI Model Server to $INSTALL_DIR"

# Create service user
echo "Creating service user..."
useradd -r -s /bin/false ai-service || true

# Detect GPU and set up environment
if [ -x "$(command -v nvidia-smi)" ]; then
  echo "NVIDIA GPU detected:"
  nvidia-smi
  
  # Check CUDA version
  CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
  echo "CUDA Version: $CUDA_VERSION"
  
  # Check for A100 GPU
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
  if [[ "$GPU_NAME" == *"A100"* ]]; then
    echo "A100 GPU detected. Configuring for optimal performance."
    IS_A100=true
  else
    echo "Warning: A100 GPU not detected. The system is optimized for A100 GPUs."
    IS_A100=false
  fi
else
  echo "Warning: NVIDIA GPU not detected. This setup requires an NVIDIA GPU."
  exit 1
fi

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y python3 python3-pip python3-venv python3-dev git wget \
  build-essential cmake libopenblas-dev libomp-dev pkg-config curl

# Create installation directory
mkdir -p $INSTALL_DIR
mkdir -p $INSTALL_DIR/models

# Set GPU memory for A100
if [ "$IS_A100" = true ]; then
  # Set a high limit for GPU memory usage
  echo "Setting high GPU memory limits for A100..."
  echo 'DefaultLimitMEMLOCK=infinity' >> /etc/systemd/system.conf
  systemctl daemon-reload
fi

# Copy files
if [ -d "./model_server" ]; then
  echo "Copying application files..."
  cp -r ./model_server $INSTALL_DIR/
  cp ./model_config.json $INSTALL_DIR/
  cp ./requirements.txt $INSTALL_DIR/
  cp ./entrypoint.sh $INSTALL_DIR/
  chmod +x $INSTALL_DIR/entrypoint.sh
else
  echo "Error: model_server directory not found!"
  exit 1
fi

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv $INSTALL_DIR/venv
source $INSTALL_DIR/venv/bin/activate
pip install --upgrade pip
pip install -r $INSTALL_DIR/requirements.txt

# Generate initial API key
echo "Generating initial API key..."
cd $INSTALL_DIR
python3 -c "
import os, sys
sys.path.append('$INSTALL_DIR')
from model_server.auth import generate_new_key
key = generate_new_key('admin', 'admin')
print(f'Generated admin API key: {key}')
with open('admin_key.txt', 'w') as f:
    f.write(key)
"

# Pre-download models
echo "Do you want to pre-download the models? This can take a significant amount of time. (y/n)"
read -r download_models
if [[ "$download_models" =~ ^[Yy]$ ]]; then
  echo "Starting model downloads..."
  
  # Extract model URLs from config
  PRIMARY_MODEL_URL=$(grep -o '"model_url": "[^"]*"' $INSTALL_DIR/model_config.json | grep -m1 -o 'https://[^"]*')
  PRIMARY_MODEL_PATH=$(grep -o '"model_path": "[^"]*"' $INSTALL_DIR/model_config.json | grep -m1 -o '"[^"]*"' | tr -d '"')
  
  CLASSIFIER_MODEL_URL=$(grep -o '"model_url": "[^"]*"' $INSTALL_DIR/model_config.json | grep -m2 | tail -n1 | grep -o 'https://[^"]*')
  CLASSIFIER_MODEL_PATH=$(grep -o '"model_path": "[^"]*"' $INSTALL_DIR/model_config.json | grep -m2 | tail -n1 | grep -o '"[^"]*"' | tr -d '"')
  
  # Create model directories
  mkdir -p $(dirname "$INSTALL_DIR/$PRIMARY_MODEL_PATH")
  mkdir -p $(dirname "$INSTALL_DIR/$CLASSIFIER_MODEL_PATH")
  
  # Download primary model
  echo "Downloading primary model from $PRIMARY_MODEL_URL"
  echo "This may take a long time due to the large file size."
  wget --progress=bar:force -O "$INSTALL_DIR/$PRIMARY_MODEL_PATH" "$PRIMARY_MODEL_URL" || echo "Download failed, will be attempted at first run"
  
  # Download classifier model
  echo "Downloading classifier model from $CLASSIFIER_MODEL_URL"
  wget --progress=bar:force -O "$INSTALL_DIR/$CLASSIFIER_MODEL_PATH" "$CLASSIFIER_MODEL_URL" || echo "Download failed, will be attempted at first run"
else
  echo "Skipping model downloads. Models will be downloaded on first use."
fi

# Set proper permissions
echo "Setting permissions..."
chown -R ai-service:ai-service $INSTALL_DIR
chmod -R 750 $INSTALL_DIR

# Create systemd service for Docker setup
if [ -f "./docker-compose.yml" ]; then
  echo "Setting up Docker service..."
  
  # Install Docker if not present
  if ! [ -x "$(command -v docker)" ]; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
  fi
  
  # Install Docker Compose if not present
  if ! [ -x "$(command -v docker-compose)" ]; then
    echo "Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
  fi
  
  # Configure Docker for NVIDIA
  if ! [ -f "/etc/docker/daemon.json" ]; then
    echo "Configuring Docker for NVIDIA GPUs..."
    mkdir -p /etc/docker
    cat > /etc/docker/daemon.json <<EOF
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
    systemctl restart docker
  fi
  
  # Copy Docker files
  cp ./docker-compose.yml $INSTALL_DIR/
  cp ./Dockerfile $INSTALL_DIR/
  
  # Create Docker service
  cat > /etc/systemd/system/model-server-docker.service <<EOF
[Unit]
Description=AI Model Server Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable model-server-docker.service
  systemctl start model-server-docker.service
  
  echo "Docker service installed and started"
else
  # Install standard systemd service
  echo "Installing systemd service..."
  
  cat > /etc/systemd/system/model-server.service <<EOF
[Unit]
Description=AI Model Server for Job Discovery
After=network.target

[Service]
User=ai-service
Group=ai-service
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python -m model_server.main
Restart=on-failure
RestartSec=5
Environment=PYTHONPATH=${PYTHONPATH}:$INSTALL_DIR
Environment=CONFIG_FILE=$INSTALL_DIR/model_config.json
Environment=WORKERS=8

# Increase limits for A100 GPU
LimitMEMLOCK=infinity
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable model-server.service
  
  # Start the service
  echo "Starting model-server service..."
  systemctl start model-server.service
  systemctl status model-server.service
fi

echo "Installation complete!"
echo "Admin API key saved to $INSTALL_DIR/admin_key.txt"
echo "Server should be accessible at http://localhost:8080"
echo ""
echo "To start, stop, or check the service:"
if [ -f "./docker-compose.yml" ]; then
  echo "  systemctl start model-server-docker.service"
  echo "  systemctl stop model-server-docker.service"
  echo "  systemctl status model-server-docker.service"
  echo ""
  echo "Or use Docker Compose directly:"
  echo "  cd $INSTALL_DIR && docker-compose up -d"
  echo "  cd $INSTALL_DIR && docker-compose down"
  echo "  cd $INSTALL_DIR && docker-compose logs -f"
else
  echo "  systemctl start model-server.service"
  echo "  systemctl stop model-server.service"
  echo "  systemctl status model-server.service"
fi
