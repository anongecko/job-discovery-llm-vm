#!/bin/bash

# Setup script for AI Model Server
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

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y python3 python3-pip python3-venv git wget

# Create installation directory
mkdir -p $INSTALL_DIR
mkdir -p $INSTALL_DIR/models

# Copy files
if [ -d "./model_server" ]; then
  echo "Copying application files..."
  cp -r ./model_server $INSTALL_DIR/
  cp ./model_config.json $INSTALL_DIR/
  cp ./requirements.txt $INSTALL_DIR/
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

# Set proper permissions
echo "Setting permissions..."
chown -R ai-service:ai-service $INSTALL_DIR
chmod -R 750 $INSTALL_DIR

# Install systemd service if available
if [ -f "./model-server.service" ]; then
  echo "Installing systemd service..."
  cp ./model-server.service /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable model-server.service
  
  # Start the service
  echo "Starting model-server service..."
  systemctl start model-server.service
  systemctl status model-server.service
else
  echo "No systemd service file found, skipping service installation."
fi

echo "Installation complete!"
echo "Admin API key saved to $INSTALL_DIR/admin_key.txt"
echo "Server should be accessible at http://localhost:8080"
echo ""
echo "To start or stop the service:"
echo "  systemctl start model-server.service"
echo "  systemctl stop model-server.service"
