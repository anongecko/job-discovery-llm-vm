# AI Model Server for Job Discovery

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)

A high-performance model server designed for AI-powered job discovery systems, optimized for A100 GPUs. This server provides powerful AI capabilities including text generation, embeddings, and classification through a consistent REST API.

## 🚀 Features

### Core Capabilities
- **Advanced Text Generation**: Using Llama 3.1 70B (Q6_K quantized) for high-quality text generation
- **Semantic Embeddings**: Fast vector embedding generation with Nomic-Embed-Text-v1
- **Text Classification**: Efficient categorization using Mistral 7B Instruct v0.2
- **Streaming Responses**: Event-stream support for real-time text generation
- **Batched Processing**: Optimized batching for embeddings and classifications

### Performance Features
- **A100 GPU Optimization**: Configured for NVIDIA A100 80GB GPU
- **Quantization**: Efficient memory usage through 4-bit and 6-bit model quantization
- **Concurrent Request Handling**: Process multiple requests simultaneously
- **Dynamic Memory Management**: Intelligent resource allocation based on workload

### Operational Features
- **Robust Authentication**: API key management with rate limiting and role-based access
- **Comprehensive Monitoring**: Detailed metrics for performance analysis
- **Health Checking**: Automated model health monitoring and maintenance
- **Docker Deployment**: Production-ready containerization
- **Auto-scaling**: Horizontal scaling support for high-load environments

## 📊 Architecture

The Model Server is built on a modern, asynchronous architecture:

```
                           ┌─────────────────┐
                           │   Client Apps   │
                           └────────┬────────┘
                                    │
                         ┌──────────▼──────────┐
┌───────────────────────┤   REST API Gateway   ├───────────────────────┐
│                        └──────────┬──────────┘                       │
│                                   │                                  │
│  ┌──────────────────┐   ┌─────────▼─────────┐   ┌──────────────────┐ │
│  │   Primary Model  │   │  Authentication   │   │  Rate Limiting   │ │
│  │  (Llama 3.3 70B) ◄───┤    & Security     ├───►      Layer      │ │
│  └────────┬─────────┘   └─────────┬─────────┘   └──────────────────┘ │
│           │                       │                                  │
│  ┌────────▼─────────┐   ┌─────────▼─────────┐   ┌──────────────────┐ │
│  │  Embedding Model │   │    Request Queue  │   │ Classification   │ │
│  │ (Nomic-Embed-Text)◄───┤   & Dispatcher   ├───►  Model (Mistral) │ │
│  └──────────────────┘   └─────────┬─────────┘   └──────────────────┘ │
│                                   │                                  │
│                        ┌──────────▼──────────┐                       │
└───────────────────────┤   Health Monitoring  ├───────────────────────┘
                         └─────────────────────┘
```

### Component Overview
- **API Gateway**: FastAPI-based HTTP interface with input validation
- **Authentication**: Token-based API key system with role permissions
- **Model Layer**: Optimized inference engines for text generation, embeddings, and classification
- **Health Monitoring**: Automatic model maintenance and resource optimization

## 🛠️ Installation

### Prerequisites
- NVIDIA GPU with CUDA 11.8+ support (A100 recommended)
- Docker and Docker Compose
- 200GB+ SSD storage
- 32GB+ system RAM

### Docker Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-model-server.git
cd ai-model-server

# Run the setup script
chmod +x setup.sh
sudo ./setup.sh

# Start the server
docker-compose up -d
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-model-server.git
cd ai-model-server

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate an API key
python -c "from model_server.auth import generate_new_key; print(generate_new_key('admin', 'admin'))"

# Start the server
python -m model_server.main
```

## ⚙️ Configuration

### Model Configuration
The server uses `model_config.json` to configure models:

```json
{
  "primary_model": {
    "model_name": "llama-3.1-70b-instruct",
    "model_type": "gguf",
    "model_path": "models/llama-3.1-70b-instruct.Q6_K.gguf",
    "model_url": "https://huggingface.co/TheBloke/Llama-3.1-70B-Instruct-GGUF/resolve/main/llama-3.1-70b-instruct.Q6_K.gguf",
    "device": "cuda",
    "n_gpu_layers": -1,
    "n_ctx": 8192,
    "max_tokens": 4096,
    "max_batch_size": 8,
    "tensor_split": "0.99,0.01",
    "threads": 12,
    "use_mlock": true,
    "n_batch": 512,
    "stream_output": true,
    "download_if_missing": true,
    "max_concurrent_requests": 12
  },
  ...
}
```

### Environment Variables
- `HOST`: Server host address (default: "0.0.0.0")
- `PORT`: Server port (default: 8080)
- `WORKERS`: Number of Uvicorn workers (default: 1)
- `CONFIG_FILE`: Path to model configuration file
- `DEBUG`: Enable debug mode ("1" or "true")
- `DOWNLOAD_MODELS_ON_START`: Download models at startup ("true" or "false")

## 🔑 Authentication

The server uses a token-based API key system:

```bash
# Generate a new API key (run in Python)
from model_server.auth import generate_new_key
new_key = generate_new_key(name="client1", role="client", rate_limit=100, expires_in_days=365)
print(new_key)
```

API keys can be managed through the `/api-keys` endpoints (admin only):
- `POST /api-keys/create`: Create a new API key
- `GET /api-keys/list`: List all API keys
- `PATCH /api-keys/update/{key}`: Update an API key
- `DELETE /api-keys/revoke/{key}`: Revoke an API key

## 📝 API Documentation

### Text Generation
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "prompt": "Extract the job title from: \"Senior Software Engineer at Google\"",
    "system_message": "You are an expert job data extractor.",
    "max_tokens": 1024,
    "temperature": 0.1,
    "stream": false
  }'
```

### Text Embeddings
```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "input": ["Software Engineer", "Data Scientist"]
  }'
```

### Text Classification
```bash
curl -X POST http://localhost:8080/v1/classifications \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "text": "Looking for a Python developer with 5+ years experience",
    "categories": ["job_posting", "resume", "cover_letter"]
  }'
```

### Streaming Text Generation
```javascript
// JavaScript example for streaming
const eventSource = new EventSource('/v1/completions?api_key=YOUR_API_KEY');
const payload = {
  prompt: "Write a job description for a Senior AI Engineer",
  stream: true,
  temperature: 0.7
};

fetch('/v1/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'YOUR_API_KEY'
  },
  body: JSON.stringify(payload)
})
.then(response => {
  const reader = response.body.getReader();
  // Process stream...
});
```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8080/health
```

### Metrics (Requires API Key)
```bash
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8080/metrics
```

The server includes Prometheus integration for comprehensive monitoring. Access Grafana dashboards at `http://localhost:3000` when using Docker deployment (default credentials: admin/model-server-admin).

## ⚡ Performance Optimizations

### A100 GPU Configuration
For optimal performance on an A100 GPU:

- **VRAM Allocation**: 
  - Primary model: ~30GB 
  - Embedding model: ~2GB 
  - Classification model: ~4GB
  - Total: ~36GB (with headroom for batching)

- **Worker Configuration**:
  - For 24 vCPUs, we recommend 8 worker processes
  - Set `WORKERS=8` in the environment or docker-compose.yml

- **Batch Processing**:
  - Embeddings support batches of up to 64 inputs
  - Use batching for processing large datasets

## 🔒 Security Considerations

- **API Key Management**:
  - Rotate keys regularly
  - Use role-based permissions
  - Set appropriate rate limits

- **Network Security**:
  - Run behind a reverse proxy for TLS
  - Use firewall rules to restrict access
  - Consider VPC deployment for production

- **Model Security**:
  - Input validation prevents prompt injection
  - Output is sanitized to prevent XSS
  - Models have safety filters enabled

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
