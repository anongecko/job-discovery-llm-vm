"""
Enhanced models implementation for the Model Server
Supports both HuggingFace and GGUF models with optimizations for A100 GPU
"""

import os
import json
import logging
import asyncio
import time
import psutil
import torch
import aiofiles
import requests
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Tuple
from pathlib import Path
import concurrent.futures
from functools import partial
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger("model-server")

# Load configuration
CONFIG_FILE = os.environ.get("CONFIG_FILE", "model_config.json")


def load_config() -> Dict[str, Any]:
    """Load model configuration from file with improved error handling."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            return config
        else:
            logger.warning(f"Config file {CONFIG_FILE} not found. Using default configuration.")
            return default_config()
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        logger.warning("Falling back to default configuration.")
        return default_config()


def default_config() -> Dict[str, Any]:
    """Default configuration for models."""
    gpu_available = torch.cuda.is_available()
    return {
        "primary_model": {
            "model_name": "llama-3.1-70b-instruct",
            "model_type": "gguf",
            "model_path": "models/llama-3.1-70b-instruct.Q6_K.gguf",
            "model_url": "https://huggingface.co/TheBloke/Llama-3.1-70B-Instruct-GGUF/resolve/main/llama-3.1-70b-instruct.Q6_K.gguf",
            "device": "cuda" if gpu_available else "cpu",
            "n_gpu_layers": -1 if gpu_available else 0,
            "n_ctx": 8192,
            "max_tokens": 4096,
            "max_batch_size": 8,
            "tensor_split": "0.99,0.01",
            "threads": 12,
            "use_mlock": True,
            "n_batch": 512,
            "download_if_missing": True,
            "max_concurrent_requests": 12,
        },
        "embedding_model": {
            "model_name": "nomic-ai/nomic-embed-text-v1",
            "model_type": "huggingface",
            "model_path": "models/nomic-embed-text-v1",
            "device": "cuda" if gpu_available else "cpu",
            "max_batch_size": 64,
            "max_concurrent_requests": 24,
            "download_if_missing": True,
        },
        "classifier_model": {
            "model_name": "mistral-7b-instruct-v0.2",
            "model_type": "gguf",
            "model_path": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "model_url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "device": "cuda" if gpu_available else "cpu",
            "n_gpu_layers": -1 if gpu_available else 0,
            "n_ctx": 4096,
            "max_tokens": 1024,
            "max_batch_size": 16,
            "threads": 6,
            "use_mlock": True,
            "n_batch": 256,
            "download_if_missing": True,
            "max_concurrent_requests": 16,
        },
    }


async def download_file(url: str, destination: str, chunk_size: int = 8192):
    """Download a file with progress tracking."""
    if os.path.exists(destination):
        logger.info(f"File already exists at {destination}")
        return destination

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Temporary file for downloading
    temp_file = f"{destination}.downloading"

    try:
        logger.info(f"Downloading {url} to {destination}")

        # Get file size
        response = requests.head(url)
        file_size = int(response.headers.get("content-length", 0))

        # Start download
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            downloaded = 0
            last_log_time = time.time()

            with open(temp_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Log progress every 5 seconds
                        current_time = time.time()
                        if current_time - last_log_time > 5:
                            progress = (downloaded / file_size) * 100 if file_size > 0 else 0
                            logger.info(f"Download progress: {progress:.2f}% ({downloaded / 1024 / 1024:.2f}MB / {file_size / 1024 / 1024:.2f}MB)")
                            last_log_time = current_time

        # Rename temp file to destination
        os.rename(temp_file, destination)
        logger.info(f"Download completed: {destination}")
        return destination

    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise


# Model base class
class BaseModel:
    def __init__(self, model_type: str, config: Dict[str, Any]):
        self.model_type = model_type
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.load_time = None
        self.max_concurrent = config.get("max_concurrent_requests", 4)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(32, self.max_concurrent * 2))
        self.last_used = time.time()
        self.initialization_lock = asyncio.Lock()

    def get_model_name(self) -> str:
        """Get model name."""
        return self.config.get("model_name", "unknown")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.is_initialized

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage of this model."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # Get GPU memory info if available
        gpu_memory = {"allocated_mb": 0, "reserved_mb": 0}
        if torch.cuda.is_available():
            try:
                gpu_memory["allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_memory["reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            except:
                pass

        return {"rss_mb": memory_info.rss / (1024 * 1024), "vms_mb": memory_info.vms / (1024 * 1024), "gpu_memory": gpu_memory}

    def update_last_used(self):
        """Update last used timestamp."""
        self.last_used = time.time()

    def time_since_last_use(self) -> float:
        """Get time since last use in seconds."""
        return time.time() - self.last_used

    async def ensure_initialized(self) -> None:
        """Ensure model is initialized, with locking to prevent multiple initializations."""
        if not self.is_initialized:
            async with self.initialization_lock:
                if not self.is_initialized:  # Check again inside lock
                    await self.initialize()

    async def initialize(self) -> None:
        """Initialize model (to be implemented by subclasses)."""
        raise NotImplementedError

    async def cleanup(self) -> None:
        """Clean up resources (to be implemented by subclasses)."""
        raise NotImplementedError


# GGUF model implementation using llama.cpp Python bindings
class GGUFModel(BaseModel):
    def __init__(self, model_type: str, config: Dict[str, Any]):
        super().__init__(model_type, config)
        self.llm = None

    async def initialize(self) -> None:
        """Initialize GGUF model using llama-cpp-python."""
        if self.is_initialized:
            return

        logger.info(f"Initializing {self.model_type} GGUF model: {self.config['model_name']}")
        start_time = time.time()

        try:
            # Ensure model file exists
            model_path = self.config["model_path"]
            if not os.path.exists(model_path) and self.config.get("download_if_missing", False):
                model_url = self.config.get("model_url")
                if not model_url:
                    raise ValueError(f"Model not found at {model_path} and no download URL provided")

                # Download model
                await download_file(model_url, model_path)

            # Run intensive loading in a separate thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._load_model)

            self.is_initialized = True
            self.load_time = time.time() - start_time
            self.update_last_used()
            logger.info(f"{self.model_type} GGUF model initialized in {self.load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_type} GGUF model: {str(e)}")
            raise

    def _load_model(self) -> None:
        """Load GGUF model in a separate thread."""
        try:
            from llama_cpp import Llama

            model_path = self.config["model_path"]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            # Configure model parameters
            n_gpu_layers = self.config.get("n_gpu_layers", -1)
            n_ctx = self.config.get("n_ctx", 4096)
            threads = self.config.get("threads", 4)
            n_batch = self.config.get("n_batch", 512)
            use_mlock = self.config.get("use_mlock", False)
            tensor_split = self.config.get("tensor_split", None)

            # Prepare tensor split if specified
            tensor_split_list = None
            if tensor_split:
                try:
                    tensor_split_list = [float(x) for x in tensor_split.split(",")]
                except:
                    logger.warning(f"Invalid tensor_split format: {tensor_split}. Using default.")

            logger.info(f"Loading GGUF model from {model_path} with n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}, threads={threads}")

            # Initialize the Llama model
            self.llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, n_batch=n_batch, use_mlock=use_mlock, n_threads=threads, tensor_split=tensor_split_list, verbose=False)

            logger.info(f"GGUF model loaded successfully: {model_path}")

        except Exception as e:
            logger.error(f"Error loading GGUF model: {str(e)}")
            raise

    async def generate_completion(
        self, prompt: str, system_message: Optional[str] = None, max_tokens: Optional[int] = None, temperature: float = 0.1, top_p: float = 0.95, stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate completion using GGUF model."""
        await self.ensure_initialized()
        self.update_last_used()

        # Prepare the prompt with system message if provided
        if system_message:
            formatted_prompt = f"<s>[INST] {system_message}\n\n{prompt} [/INST]"
        else:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"

        # Get max tokens
        max_tokens_val = max_tokens or self.config.get("max_tokens", 1024)

        # Acquire semaphore for resource limiting
        async with self.semaphore:
            try:
                if stream:
                    return self._generate_stream(formatted_prompt, max_tokens_val, temperature, top_p)
                else:
                    # Run generation in a separate thread
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.executor, lambda: self._generate(formatted_prompt, max_tokens_val, temperature, top_p))
                    return result
            except Exception as e:
                logger.error(f"Error generating completion: {str(e)}")
                raise

    def _generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        """Run generation in a separate thread."""
        try:
            # Configure generation parameters
            generation_config = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "echo": False,
            }

            # Generate completion
            result = self.llm(prompt, **generation_config)

            # Extract and return the generated text
            generated_text = result["choices"][0]["text"].strip()
            return generated_text

        except Exception as e:
            logger.error(f"Error in _generate: {str(e)}")
            raise

    async def _generate_stream(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> AsyncGenerator[str, None]:
        """Stream generation results."""
        try:
            # Configure generation parameters
            generation_config = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "echo": False,
                "stream": True,
            }

            # Create a partial function for streaming generation
            generate_func = partial(self.llm, prompt, **generation_config)

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            stream_gen = await loop.run_in_executor(self.executor, generate_func)

            # Yield generated tokens
            for chunk in stream_gen:
                token = chunk["choices"][0]["text"]
                if token:
                    yield token

        except Exception as e:
            logger.error(f"Error in _generate_stream: {str(e)}")
            raise

    async def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text into categories using the GGUF model."""
        await self.ensure_initialized()
        self.update_last_used()

        async with self.semaphore:
            try:
                # Run classification in a separate thread
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, lambda: self._classify(text, categories))
                return result
            except Exception as e:
                logger.error(f"Error classifying text: {str(e)}")
                raise

    def _classify(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Run classification in a separate thread."""
        try:
            import json

            # Construct a clear and concise prompt
            categories_str = ", ".join([f'"{cat}"' for cat in categories])
            prompt = f"""<s>[INST]Classify the following text into these categories: {categories_str}

Text to classify: "{text}"

For each category, provide a confidence score between 0.0 and 1.0.
Return only a valid JSON object with categories as keys and confidence scores as values.
Only include categories that apply with non-zero scores.

Example response format:
{{
  "category1": 0.9,
  "category3": 0.4
}}
[/INST]"""

            # Configure generation parameters
            generation_config = {
                "max_tokens": 250,  # Short response is sufficient for JSON
                "temperature": 0.1,  # Low temperature for deterministic output
                "top_p": 0.95,
                "echo": False,
            }

            # Generate completion
            result = self.llm(prompt, **generation_config)
            output_text = result["choices"][0]["text"].strip()

            # Extract the JSON object
            try:
                # Try to parse JSON directly
                result_dict = json.loads(output_text)

                # Validate and clean results
                cleaned_result = {}
                for category in categories:
                    if category in result_dict:
                        try:
                            score = float(result_dict[category])
                            if 0 <= score <= 1:
                                cleaned_result[category] = score
                        except (ValueError, TypeError):
                            pass

                # Ensure at least one category is returned
                if not cleaned_result and categories:
                    cleaned_result = {categories[0]: 1.0}

                return cleaned_result

            except json.JSONDecodeError:
                # Fallback: try to extract JSON using regex
                import re

                json_match = re.search(r"{.*}", output_text, re.DOTALL)
                if json_match:
                    try:
                        extracted_json = json_match.group(0)
                        result_dict = json.loads(extracted_json)

                        # Clean results
                        cleaned_result = {}
                        for category in categories:
                            if category in result_dict:
                                try:
                                    score = float(result_dict[category])
                                    if 0 <= score <= 1:
                                        cleaned_result[category] = score
                                except (ValueError, TypeError):
                                    pass

                        if cleaned_result:
                            return cleaned_result
                    except:
                        pass

            # Emergency fallback
            logger.warning(f"Failed to parse classification result as JSON: {output_text}")
            fallback_result = {category: 1.0 / len(categories) for category in categories}
            return fallback_result

        except Exception as e:
            logger.error(f"Error in _classify: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Clean up GGUF model resources."""
        if not self.is_initialized:
            return

        logger.info(f"Cleaning up {self.model_type} GGUF model resources")

        try:
            # Free up resources
            if self.llm is not None:
                del self.llm
                self.llm = None

            # Force garbage collection
            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_initialized = False
            logger.info(f"{self.model_type} GGUF model resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up {self.model_type} GGUF model: {str(e)}")


# HuggingFace embedding model implementation
class EmbeddingModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("embedding", config)
        self.model_type = config.get("model_type", "huggingface")

    async def initialize(self) -> None:
        """Initialize embedding model."""
        if self.is_initialized:
            return

        logger.info(f"Initializing embedding model: {self.config['model_name']}")
        start_time = time.time()

        try:
            # Run intensive loading in a separate thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._load_model)

            self.is_initialized = True
            self.load_time = time.time() - start_time
            self.update_last_used()
            logger.info(f"Embedding model initialized in {self.load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

    def _load_model(self) -> None:
        """Load embedding model in a separate thread."""
        try:
            model_name = self.config["model_name"]
            model_path = self.config["model_path"]
            device = self.config["device"]

            # Check if using Nomic embedding model
            if "nomic" in model_name.lower():
                try:
                    import nomic
                    from nomic.embed import EmbeddingModel as NomicEmbedder

                    logger.info(f"Initializing Nomic embedding model: {model_name}")
                    self.model = NomicEmbedder(model_name=model_name)
                    logger.info("Nomic embedding model loaded successfully")
                    return
                except ImportError:
                    logger.warning("Nomic package not found, falling back to sentence-transformers")

            # Default to sentence-transformers
            from sentence_transformers import SentenceTransformer

            # Check if model is already downloaded
            if os.path.exists(model_path) and os.path.isdir(model_path):
                # Load model from local path
                logger.info(f"Loading embedding model from local path: {model_path}")
                load_path = model_path
            else:
                # Download model if missing and allowed
                if self.config.get("download_if_missing", False):
                    logger.info(f"Downloading embedding model: {model_name}")
                    load_path = model_name
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                else:
                    raise FileNotFoundError(f"Model not found at {model_path} and download_if_missing is False")

            # Load the model
            self.model = SentenceTransformer(load_path, device=device)

            # Save model locally if downloaded
            if load_path == model_name and not os.path.exists(model_path):
                logger.info(f"Saving embedding model to {model_path}")
                self.model.save(model_path)

        except Exception as e:
            logger.error(f"Error in _load_model: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts with retry logic."""
        await self.ensure_initialized()
        self.update_last_used()

        # Split texts into batches based on max_batch_size
        max_batch_size = self.config.get("max_batch_size", 64)
        batches = [texts[i : i + max_batch_size] for i in range(0, len(texts), max_batch_size)]
        all_embeddings = []

        async with self.semaphore:
            for batch in batches:
                try:
                    # Run embedding generation in a separate thread
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.executor, lambda: self._embed(batch))
                    all_embeddings.extend(result)
                except Exception as e:
                    logger.error(f"Error generating embeddings: {str(e)}")
                    raise

        return all_embeddings

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Run embedding in a separate thread."""
        try:
            # Check if using Nomic embedding model
            if hasattr(self.model, "embed"):
                # This is the Nomic embedding model
                embeddings = self.model.embed(texts)
            else:
                # This is the sentence-transformers model
                embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)

            # Convert to list format if necessary
            if not isinstance(embeddings, list):
                embeddings = embeddings.tolist()

            # Ensure all embeddings are properly formatted
            result = []
            for emb in embeddings:
                if isinstance(emb, list):
                    result.append(emb)
                else:
                    result.append(emb.tolist())

            return result
        except Exception as e:
            logger.error(f"Error in _embed: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Clean up embedding model resources."""
        if not self.is_initialized:
            return

        logger.info("Cleaning up embedding model resources")

        try:
            if self.model is not None:
                del self.model
                self.model = None

            # Force garbage collection
            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_initialized = False
            logger.info("Embedding model resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up embedding model: {str(e)}")


# Load configuration
config = load_config()

# Create model instances based on model type
primary_model = GGUFModel("primary", config["primary_model"])
embedding_model = EmbeddingModel(config["embedding_model"])
classifier_model = GGUFModel("classifier", config["classifier_model"])


async def initialize_models():
    """Initialize all models with prioritization for improved startup time."""
    # Start with embedding model as it's typically the smallest
    await embedding_model.initialize()

    # Initialize classifier model next
    await classifier_model.initialize()

    # Initialize primary model last as it's the largest
    await primary_model.initialize()

    logger.info("All models initialized")


async def cleanup_models():
    """Clean up all model resources."""
    cleanup_tasks = []

    # Clean up in reverse order of importance
    if classifier_model.is_initialized:
        cleanup_tasks.append(classifier_model.cleanup())

    if embedding_model.is_initialized:
        cleanup_tasks.append(embedding_model.cleanup())

    if primary_model.is_initialized:
        cleanup_tasks.append(primary_model.cleanup())

    # Wait for all cleanup tasks to complete
    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks)

    logger.info("All model resources cleaned up")


async def check_model_health():
    """Check model health and perform maintenance if needed."""
    # Check if any models need to be unloaded due to memory pressure
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        memory_utilization = allocated_memory / total_memory

        # If memory utilization is high (>90%), consider unloading models
        if memory_utilization > 0.9:
            logger.warning(f"High GPU memory utilization: {memory_utilization:.2f}. Considering maintenance.")

            # Find least recently used model to unload
            models = [primary_model, embedding_model, classifier_model]
            loaded_models = [m for m in models if m.is_initialized]

            if loaded_models:
                # Sort by last used time
                loaded_models.sort(key=lambda m: m.last_used)

                # Unload least recently used model if it hasn't been used in a while
                oldest_model = loaded_models[0]
                if oldest_model.time_since_last_use() > 300:  # 5 minutes
                    logger.info(f"Unloading {oldest_model.model_type} model due to memory pressure")
                    await oldest_model.cleanup()

