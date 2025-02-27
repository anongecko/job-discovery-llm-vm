"""
Model Server for Job Discovery System
Provides API endpoints for all three model types
"""

import os
import logging
import secrets
import json
import time
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger("model-server")

# Create FastAPI app
app = FastAPI(title="AI Model Server", description="API for AI models used by the Job Discovery System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import model modules
from model_server.models import primary_model, embedding_model, classifier_model, initialize_models, cleanup_models

from model_server.auth import validate_api_key


# Request and response models
class CompletionRequest(BaseModel):
    prompt: str
    system_message: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.1
    top_p: float = 0.95


class CompletionResponse(BaseModel):
    text: str
    model: str
    processing_time: float


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    processing_time: float


class ClassificationRequest(BaseModel):
    text: str
    categories: List[str]


class ClassificationResponse(BaseModel):
    classifications: Dict[str, float]
    model: str
    processing_time: float


# Metrics tracking
request_count = 0
error_count = 0
model_usage = {"primary": 0, "embedding": 0, "classifier": 0}


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting model server...")
    await initialize_models()
    logger.info("Model server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down model server...")
    await cleanup_models()
    logger.info("Model server shut down successfully")


# Routes
@app.get("/")
async def root():
    return {"name": "AI Model Server", "status": "online", "models": {"primary": primary_model.is_loaded(), "embedding": embedding_model.is_loaded(), "classifier": classifier_model.is_loaded()}}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/metrics")
async def get_metrics(api_key: str = Depends(validate_api_key)):
    return {
        "request_count": request_count,
        "error_count": error_count,
        "model_usage": model_usage,
        "memory_usage": {"primary": primary_model.get_memory_usage(), "embedding": embedding_model.get_memory_usage(), "classifier": classifier_model.get_memory_usage()},
    }


# Primary model route
@app.post("/v1/completions", response_model=CompletionResponse)
async def generate_completion(request: CompletionRequest, api_key: str = Depends(validate_api_key)):
    global request_count, model_usage
    request_count += 1
    model_usage["primary"] += 1

    start_time = time.time()
    try:
        text = await primary_model.generate_completion(
            prompt=request.prompt, system_message=request.system_message, max_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p
        )
        processing_time = time.time() - start_time

        return {"text": text, "model": primary_model.get_model_name(), "processing_time": processing_time}
    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}")
        global error_count
        error_count += 1
        raise HTTPException(status_code=500, detail=str(e))


# Embedding model route
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest, api_key: str = Depends(validate_api_key)):
    global request_count, model_usage
    request_count += 1
    model_usage["embedding"] += 1

    start_time = time.time()
    try:
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input

        embeddings = await embedding_model.generate_embeddings(texts)
        processing_time = time.time() - start_time

        return {"embeddings": embeddings, "model": embedding_model.get_model_name(), "processing_time": processing_time}
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        global error_count
        error_count += 1
        raise HTTPException(status_code=500, detail=str(e))


# Classifier model route
@app.post("/v1/classifications", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest, api_key: str = Depends(validate_api_key)):
    global request_count, model_usage
    request_count += 1
    model_usage["classifier"] += 1

    start_time = time.time()
    try:
        classifications = await classifier_model.classify_text(text=request.text, categories=request.categories)
        processing_time = time.time() - start_time

        return {"classifications": classifications, "model": classifier_model.get_model_name(), "processing_time": processing_time}
    except Exception as e:
        logger.error(f"Error classifying text: {str(e)}")
        global error_count
        error_count += 1
        raise HTTPException(status_code=500, detail=str(e))


# Middleware for logging and error handling
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"{request.method} {request.url.path} completed in {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"{request.method} {request.url.path} failed in {process_time:.4f}s: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Internal server error: {str(e)}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("model_server.main:app", host="0.0.0.0", port=8080, reload=False, workers=1)
