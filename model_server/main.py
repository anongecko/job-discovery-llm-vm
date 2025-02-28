"""
Enhanced Model Server for Job Discovery System
Provides API endpoints with streaming capabilities and improved performance for A100 GPU
"""

import os
import logging
import time
import asyncio
import json
import psutil
from typing import List, Dict, Any, Optional, Union, AsyncIterator

from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field, root_validator, validator
import uvicorn

# Set up logging with improved formatting
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger("model-server")

# Create FastAPI app
app = FastAPI(title="AI Model Server", description="Enhanced API for AI models used by the Job Discovery System", version="2.0.0", docs_url="/docs", redoc_url="/redoc")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import model modules
from model_server.models import primary_model, embedding_model, classifier_model, initialize_models, cleanup_models, check_model_health

# Import auth module with enhanced API key validation
from model_server.auth import validate_api_key

# Import API key management router
from model_server.api_keys import router as api_keys_router

# Include API key management routes
app.include_router(api_keys_router)


# Request and response models with enhanced validation
class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to generate completion for")
    system_message: Optional[str] = Field(None, description="Optional system message for instruction")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature (0.0 to 2.0)")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling parameter (0.0 to 1.0)")
    stream: bool = Field(False, description="Whether to stream the response")

    @validator("prompt")
    def prompt_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()


class CompletionResponse(BaseModel):
    text: str = Field(..., description="Generated completion text")
    model: str = Field(..., description="Model name used for generation")
    processing_time: float = Field(..., description="Time taken to generate completion in seconds")
    total_tokens: Optional[int] = Field(None, description="Total tokens in prompt and completion")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text(s) to generate embeddings for")

    @validator("input")
    def input_not_empty(cls, v):
        if isinstance(v, str):
            if not v or not v.strip():
                raise ValueError("Input text cannot be empty")
            return v.strip()
        elif isinstance(v, list):
            if not v:
                raise ValueError("Input list cannot be empty")
            if any(not text or not text.strip() for text in v if isinstance(text, str)):
                raise ValueError("All input texts must be non-empty")
            return [text.strip() if isinstance(text, str) else text for text in v]
        return v


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model name used for embedding")
    processing_time: float = Field(..., description="Time taken to generate embeddings in seconds")
    dimensions: int = Field(..., description="Dimensions of the embeddings")


class ClassificationRequest(BaseModel):
    text: str = Field(..., description="Text to classify")
    categories: List[str] = Field(..., description="Categories to classify into")

    @validator("text")
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

    @validator("categories")
    def categories_not_empty(cls, v):
        if not v:
            raise ValueError("Categories list cannot be empty")
        if any(not cat or not cat.strip() for cat in v if isinstance(cat, str)):
            raise ValueError("All categories must be non-empty")
        return [cat.strip() if isinstance(cat, str) else cat for cat in v]


class ClassificationResponse(BaseModel):
    classifications: Dict[str, float] = Field(..., description="Classification results with confidence scores")
    model: str = Field(..., description="Model name used for classification")
    processing_time: float = Field(..., description="Time taken to classify in seconds")


# Metrics tracking with atomic updates
class Metrics:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.model_usage = {"primary": 0, "embedding": 0, "classifier": 0}
        self.average_latency = {"primary": 0, "embedding": 0, "classifier": 0}
        self.request_count_per_model = {"primary": 0, "embedding": 0, "classifier": 0}
        self.lock = asyncio.Lock()

    async def increment_request(self, model_type: str = None):
        async with self.lock:
            self.request_count += 1
            if model_type:
                self.model_usage[model_type] = self.model_usage.get(model_type, 0) + 1

    async def increment_error(self):
        async with self.lock:
            self.error_count += 1

    async def update_latency(self, model_type: str, latency: float):
        async with self.lock:
            current_count = self.request_count_per_model.get(model_type, 0)
            current_avg = self.average_latency.get(model_type, 0)

            # Update moving average
            new_count = current_count + 1
            new_avg = ((current_avg * current_count) + latency) / new_count

            self.request_count_per_model[model_type] = new_count
            self.average_latency[model_type] = new_avg

    async def get_metrics(self):
        async with self.lock:
            return {"request_count": self.request_count, "error_count": self.error_count, "model_usage": dict(self.model_usage), "average_latency": dict(self.average_latency)}


metrics = Metrics()


# Model health monitoring
async def schedule_health_checks():
    while True:
        try:
            await check_model_health()
        except Exception as e:
            logger.error(f"Error in health check: {e}")

        # Run health check every 5 minutes
        await asyncio.sleep(300)


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting enhanced model server...")

    # Initialize models
    await initialize_models()

    # Start health check task
    asyncio.create_task(schedule_health_checks())

    logger.info("Model server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down model server...")
    await cleanup_models()
    logger.info("Model server shut down successfully")


# Routes
@app.get("/")
async def root():
    """Get basic server information"""
    return {
        "name": "Enhanced AI Model Server for Job Discovery",
        "version": "2.0.0",
        "status": "online",
        "models": {
            "primary": {"loaded": primary_model.is_loaded(), "name": primary_model.get_model_name()},
            "embedding": {"loaded": embedding_model.is_loaded(), "name": embedding_model.get_model_name()},
            "classifier": {"loaded": classifier_model.is_loaded(), "name": classifier_model.get_model_name()},
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Basic health check
    health_info = {"status": "healthy"}

    # Add model information
    model_info = {}
    for name, model in [("primary", primary_model), ("embedding", embedding_model), ("classifier", classifier_model)]:
        model_info[name] = {"loaded": model.is_loaded(), "last_used": int(time.time() - model.last_used) if model.is_loaded() else None}

    health_info["models"] = model_info

    # Add system information
    health_info["system"] = {"cpu_percent": psutil.cpu_percent(), "memory_percent": psutil.virtual_memory().percent}

    # Add GPU information if available
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            }
            health_info["gpu"] = gpu_info
    except:
        health_info["gpu"] = {"error": "Failed to get GPU information"}

    return health_info


@app.get("/metrics")
async def get_metrics(request: Request, api_key: str = Depends(validate_api_key)):
    """Get detailed server metrics"""
    # Get basic metrics
    base_metrics = await metrics.get_metrics()

    # Add memory usage
    memory_usage = {
        "primary": primary_model.get_memory_usage() if primary_model.is_loaded() else None,
        "embedding": embedding_model.get_memory_usage() if embedding_model.is_loaded() else None,
        "classifier": classifier_model.get_memory_usage() if classifier_model.is_loaded() else None,
    }

    # Add system metrics
    system_metrics = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory": {
            "total_gb": psutil.virtual_memory().total / (1024 * 1024 * 1024),
            "available_gb": psutil.virtual_memory().available / (1024 * 1024 * 1024),
            "percent": psutil.virtual_memory().percent,
        },
        "disk": {"total_gb": psutil.disk_usage("/").total / (1024 * 1024 * 1024), "free_gb": psutil.disk_usage("/").free / (1024 * 1024 * 1024), "percent": psutil.disk_usage("/").percent},
    }

    return {**base_metrics, "memory_usage": memory_usage, "system": system_metrics}


# Primary model route with streaming support
@app.post("/v1/completions", response_model=CompletionResponse)
async def generate_completion(request: Request, request_data: CompletionRequest, api_key: str = Depends(validate_api_key), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Generate text completion with optional streaming"""
    # Record request in metrics
    await metrics.increment_request("primary")

    start_time = time.time()

    # Handle streaming response if requested
    if request_data.stream:
        return await stream_completion(request_data, api_key, start_time, background_tasks)

    try:
        # Generate text
        text = await primary_model.generate_completion(
            prompt=request_data.prompt, system_message=request_data.system_message, max_tokens=request_data.max_tokens, temperature=request_data.temperature, top_p=request_data.top_p, stream=False
        )

        processing_time = time.time() - start_time

        # Record metrics
        await metrics.update_latency("primary", processing_time)

        # Estimate token count
        total_tokens = None
        try:
            import re

            # Rough estimate: 4 chars per token on average
            prompt_tokens = len(request_data.prompt) // 4
            completion_tokens = len(text) // 4
            total_tokens = prompt_tokens + completion_tokens
        except:
            pass

        return {"text": text, "model": primary_model.get_model_name(), "processing_time": processing_time, "total_tokens": total_tokens}
    except Exception as e:
        # Record error
        await metrics.increment_error()
        logger.error(f"Error generating completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_completion(request: CompletionRequest, api_key: str, start_time: float, background_tasks: BackgroundTasks):
    """Stream completion as server-sent events"""
    try:
        # Generate streaming response
        generator = primary_model.generate_completion(
            prompt=request.prompt, system_message=request.system_message, max_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p, stream=True
        )

        async def event_generator():
            try:
                full_text = ""
                async for token in generator:
                    full_text += token
                    # Yield JSON event
                    data = json.dumps({"text": token, "full_text": full_text})
                    yield f"data: {data}\n\n"

                # Final event with timing information
                processing_time = time.time() - start_time
                await metrics.update_latency("primary", processing_time)

                data = json.dumps({"text": "", "full_text": full_text, "model": primary_model.get_model_name(), "processing_time": processing_time, "finished": True})
                yield f"data: {data}\n\n"

            except Exception as e:
                await metrics.increment_error()
                logger.error(f"Error in stream generation: {str(e)}")
                # Send error event
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"

        return EventSourceResponse(event_generator())

    except Exception as e:
        await metrics.increment_error()
        logger.error(f"Error setting up streaming: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Embedding model route with batching optimizations
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: Request, request_data: EmbeddingRequest, api_key: str = Depends(validate_api_key)):
    """Generate text embeddings with improved batching"""
    # Record request in metrics
    await metrics.increment_request("embedding")

    start_time = time.time()

    try:
        # Convert input to list if it's a string
        texts = [request_data.input] if isinstance(request_data.input, str) else request_data.input

        # Generate embeddings
        embeddings = await embedding_model.generate_embeddings(texts)

        processing_time = time.time() - start_time

        # Record metrics
        await metrics.update_latency("embedding", processing_time)

        # Get dimensions
        dimensions = len(embeddings[0]) if embeddings and embeddings[0] else 0

        return {"embeddings": embeddings, "model": embedding_model.get_model_name(), "processing_time": processing_time, "dimensions": dimensions}
    except Exception as e:
        # Record error
        await metrics.increment_error()
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Classifier model route
@app.post("/v1/classifications", response_model=ClassificationResponse)
async def classify_text(request: Request, request_data: ClassificationRequest, api_key: str = Depends(validate_api_key)):
    """Classify text into categories"""
    # Record request in metrics
    await metrics.increment_request("classifier")

    start_time = time.time()

    try:
        # Generate classifications
        classifications = await classifier_model.classify_text(text=request_data.text, categories=request_data.categories)

        processing_time = time.time() - start_time

        # Record metrics
        await metrics.update_latency("classifier", processing_time)

        return {"classifications": classifications, "model": classifier_model.get_model_name(), "processing_time": processing_time}
    except Exception as e:
        # Record error
        await metrics.increment_error()
        logger.error(f"Error classifying text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Middleware for logging and error handling
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests and handle errors"""
    start_time = time.time()

    # Get client info
    client = request.client.host if request.client else "unknown"

    # Log request
    logger.info(f"Request started: {request.method} {request.url.path} from {client}")

    try:
        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Add timing header
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        # Log completion
        status_code = response.status_code
        logger.info(f"Request completed: {request.method} {request.url.path} from {client} - {status_code} in {process_time:.4f}s")

        return response
    except Exception as e:
        # Calculate time even for errors
        process_time = time.time() - start_time

        # Log error
        logger.error(f"Request failed: {request.method} {request.url.path} from {client} in {process_time:.4f}s - {str(e)}")

        # Return error response
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {str(e)}"})


if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    workers = int(os.environ.get("WORKERS", 1))

    # Configure Uvicorn options based on environment
    uvicorn_config = {"host": host, "port": port, "log_level": "info", "timeout_keep_alive": 120, "workers": workers, "reload": False}

    # If workers > 1, we can't use reload as it breaks the process model
    if workers > 1:
        logger.info(f"Starting with {workers} workers")
        uvicorn.run("model_server.main:app", **uvicorn_config)
    else:
        logger.info("Starting in single worker mode")
        uvicorn_config["reload"] = os.environ.get("DEBUG", "").lower() in ("1", "true")
        uvicorn.run(app, **uvicorn_config)

