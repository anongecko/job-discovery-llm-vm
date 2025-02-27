"""
Models implementation for the Model Server
Loads and manages models for inference
"""

import os
import json
import logging
import asyncio
import time
import psutil
import torch
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import concurrent.futures

# Configure logging
logger = logging.getLogger("model-server")

# Load configuration
CONFIG_FILE = os.environ.get("CONFIG_FILE", "model_config.json")


def load_config() -> Dict[str, Any]:
    """Load model configuration from file."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            return config
        else:
            # Return default configuration
            return {
                "primary_model": {
                    "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "model_path": "models/mixtral",
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "quantization": "4bit" if torch.cuda.is_available() else None,
                    "max_tokens": 1024,
                    "max_batch_size": 4,
                    "download_if_missing": True,
                },
                "embedding_model": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "model_path": "models/all-MiniLM-L6-v2",
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "download_if_missing": True,
                },
                "classifier_model": {
                    "model_name": "google/gemma-2b-it",
                    "model_path": "models/gemma-2b",
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "quantization": "4bit" if torch.cuda.is_available() else None,
                    "max_tokens": 256,
                    "download_if_missing": True,
                },
            }
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
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
        self.semaphore = asyncio.Semaphore(config.get("max_batch_size", 4))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

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
        return {"rss_mb": memory_info.rss / (1024 * 1024), "vms_mb": memory_info.vms / (1024 * 1024)}

    async def initialize(self) -> None:
        """Initialize model (to be implemented by subclasses)."""
        raise NotImplementedError

    async def cleanup(self) -> None:
        """Clean up resources (to be implemented by subclasses)."""
        raise NotImplementedError


# Primary model implementation
class PrimaryModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("primary", config)

    async def initialize(self) -> None:
        """Initialize primary model."""
        if self.is_initialized:
            return

        logger.info(f"Initializing primary model: {self.config['model_name']}")
        start_time = time.time()

        try:
            # Run intensive loading in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._load_model)

            self.is_initialized = True
            self.load_time = time.time() - start_time
            logger.info(f"Primary model initialized in {self.load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize primary model: {str(e)}")
            raise

    def _load_model(self) -> None:
        """Load model in a separate thread."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_name = self.config["model_name"]
            model_path = self.config["model_path"]
            device = self.config["device"]
            quantization = self.config.get("quantization")

            # Check if model is already downloaded
            if os.path.exists(model_path) and os.path.isdir(model_path):
                # Load model from local path
                logger.info(f"Loading primary model from local path: {model_path}")
                load_path = model_path
            else:
                # Download model if missing and allowed
                if self.config.get("download_if_missing", False):
                    logger.info(f"Downloading primary model: {model_name}")
                    load_path = model_name
                    os.makedirs(model_path, exist_ok=True)
                else:
                    raise FileNotFoundError(f"Model not found at {model_path} and download_if_missing is False")

            # Configure loading options based on quantization
            if quantization == "4bit":
                logger.info("Using 4-bit quantization for primary model")
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)

                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

                self.model = AutoModelForCausalLM.from_pretrained(load_path, device_map="auto", quantization_config=quantization_config, torch_dtype=torch.float16)
            elif quantization == "8bit":
                logger.info("Using 8-bit quantization for primary model")
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
                self.model = AutoModelForCausalLM.from_pretrained(load_path, device_map="auto", load_in_8bit=True)
            else:
                # No quantization
                logger.info(f"Loading primary model without quantization to device: {device}")
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
                self.model = AutoModelForCausalLM.from_pretrained(load_path).to(device)

            # Save model locally if downloaded
            if load_path == model_name and not os.path.exists(model_path):
                logger.info(f"Saving primary model to {model_path}")
                self.tokenizer.save_pretrained(model_path)
                self.model.save_pretrained(model_path)

        except Exception as e:
            logger.error(f"Error in _load_model: {str(e)}")
            raise

    async def generate_completion(self, prompt: str, system_message: Optional[str] = None, max_tokens: Optional[int] = None, temperature: float = 0.1, top_p: float = 0.95) -> str:
        """Generate completion using the primary model."""
        if not self.is_initialized:
            await self.initialize()

        async with self.semaphore:
            try:
                # Run generation in a separate thread to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, lambda: self._generate(prompt, system_message, max_tokens, temperature, top_p))
                return result
            except Exception as e:
                logger.error(f"Error generating completion: {str(e)}")
                raise

    def _generate(self, prompt: str, system_message: Optional[str], max_tokens: Optional[int], temperature: float, top_p: float) -> str:
        """Run generation in a separate thread."""
        try:
            # Prepare input
            if system_message:
                messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]

                # Format for chat models
                chat_template = self.tokenizer.chat_template
                if chat_template:
                    formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    # Fallback to manual formatting
                    if system_message:
                        formatted_prompt = f"<s>[INST] {system_message}\n\n{prompt} [/INST]"
                    else:
                        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt

            # Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.model.device)

            # Generate
            with torch.no_grad():
                generation_config = {
                    "max_new_tokens": max_tokens or self.config.get("max_tokens", 1024),
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }

                outputs = self.model.generate(input_ids, **generation_config)

            # Decode and return
            completion = self.tokenizer.decode(outputs[0][len(input_ids[0]) :], skip_special_tokens=True)
            return completion.strip()

        except Exception as e:
            logger.error(f"Error in _generate: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Clean up primary model resources."""
        try:
            # Free up GPU memory
            if self.model is not None:
                del self.model
                self.model = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Force garbage collection
            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_initialized = False
            logger.info("Primary model resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up primary model: {str(e)}")


# Embedding model implementation
class EmbeddingModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("embedding", config)

    async def initialize(self) -> None:
        """Initialize embedding model."""
        if self.is_initialized:
            return

        logger.info(f"Initializing embedding model: {self.config['model_name']}")
        start_time = time.time()

        try:
            # Run intensive loading in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._load_model)

            self.is_initialized = True
            self.load_time = time.time() - start_time
            logger.info(f"Embedding model initialized in {self.load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

    def _load_model(self) -> None:
        """Load model in a separate thread."""
        try:
            from sentence_transformers import SentenceTransformer

            model_name = self.config["model_name"]
            model_path = self.config["model_path"]
            device = self.config["device"]

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

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        if not self.is_initialized:
            await self.initialize()

        async with self.semaphore:
            try:
                # Run embedding generation in a separate thread
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, lambda: self._embed(texts))
                return result
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                raise

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Run embedding in a separate thread."""
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts)

            # Convert to list format if necessary
            if not isinstance(embeddings, list):
                embeddings = embeddings.tolist()

            return embeddings
        except Exception as e:
            logger.error(f"Error in _embed: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Clean up embedding model resources."""
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


# Classifier model implementation
class ClassifierModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("classifier", config)

    async def initialize(self) -> None:
        """Initialize classifier model."""
        if self.is_initialized:
            return

        logger.info(f"Initializing classifier model: {self.config['model_name']}")
        start_time = time.time()

        try:
            # Run intensive loading in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._load_model)

            self.is_initialized = True
            self.load_time = time.time() - start_time
            logger.info(f"Classifier model initialized in {self.load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to initialize classifier model: {str(e)}")
            raise

    def _load_model(self) -> None:
        """Load model in a separate thread."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_name = self.config["model_name"]
            model_path = self.config["model_path"]
            device = self.config["device"]
            quantization = self.config.get("quantization")

            # Check if model is already downloaded
            if os.path.exists(model_path) and os.path.isdir(model_path):
                # Load model from local path
                logger.info(f"Loading classifier model from local path: {model_path}")
                load_path = model_path
            else:
                # Download model if missing and allowed
                if self.config.get("download_if_missing", False):
                    logger.info(f"Downloading classifier model: {model_name}")
                    load_path = model_name
                    os.makedirs(model_path, exist_ok=True)
                else:
                    raise FileNotFoundError(f"Model not found at {model_path} and download_if_missing is False")

            # Configure loading options based on quantization
            if quantization == "4bit":
                logger.info("Using 4-bit quantization for classifier model")
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)

                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

                self.model = AutoModelForCausalLM.from_pretrained(load_path, device_map="auto", quantization_config=quantization_config, torch_dtype=torch.float16)
            elif quantization == "8bit":
                logger.info("Using 8-bit quantization for classifier model")
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
                self.model = AutoModelForCausalLM.from_pretrained(load_path, device_map="auto", load_in_8bit=True)
            else:
                # No quantization
                logger.info(f"Loading classifier model without quantization to device: {device}")
                self.tokenizer = AutoTokenizer.from_pretrained(load_path)
                self.model = AutoModelForCausalLM.from_pretrained(load_path).to(device)

            # Save model locally if downloaded
            if load_path == model_name and not os.path.exists(model_path):
                logger.info(f"Saving classifier model to {model_path}")
                self.tokenizer.save_pretrained(model_path)
                self.model.save_pretrained(model_path)

        except Exception as e:
            logger.error(f"Error in _load_model: {str(e)}")
            raise

    async def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text into categories."""
        if not self.is_initialized:
            await self.initialize()

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

            # Construct the prompt
            prompt = f"""
            Classify the following text into one or more of these categories: {", ".join(categories)}
            
            Text to classify: "{text}"

            For each applicable category, provide a confidence score between 0.0 and 1.0.
            Return your response in valid JSON format with categories as keys and confidence scores as values.
            Only include categories with non-zero confidence scores.
            """

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.model.device)

            # Generate
            with torch.no_grad():
                generation_config = {
                    "max_new_tokens": self.config.get("max_tokens", 256),
                    "temperature": 0.1,  # Low temperature for deterministic output
                    "top_p": 0.95,
                    "do_sample": False,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }

                outputs = self.model.generate(input_ids, **generation_config)

            # Decode and extract JSON
            output_text = self.tokenizer.decode(outputs[0][len(input_ids[0]) :], skip_special_tokens=True)

            # Try to parse JSON from the output
            try:
                # Try direct JSON parsing
                result = json.loads(output_text.strip())

                # Validate result
                if not isinstance(result, dict):
                    raise ValueError("Output is not a dictionary")

                # Convert any non-float values to floats and filter out unwanted keys
                cleaned_result = {}
                for category in categories:
                    if category in result:
                        try:
                            score = float(result[category])
                            if 0 <= score <= 1:
                                cleaned_result[category] = score
                        except (ValueError, TypeError):
                            pass

                return cleaned_result

            except json.JSONDecodeError:
                # Try to extract JSON using regex
                import re

                json_match = re.search(r"{.*}", output_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))

                        # Validate and clean result
                        if not isinstance(result, dict):
                            raise ValueError("Extracted JSON is not a dictionary")

                        cleaned_result = {}
                        for category in categories:
                            if category in result:
                                try:
                                    score = float(result[category])
                                    if 0 <= score <= 1:
                                        cleaned_result[category] = score
                                except (ValueError, TypeError):
                                    pass

                        return cleaned_result
                    except:
                        pass

            # Fallback: assign equal probabilities to each category
            return {category: 1.0 / len(categories) for category in categories}

        except Exception as e:
            logger.error(f"Error in _classify: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Clean up classifier model resources."""
        try:
            if self.model is not None:
                del self.model
                self.model = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Force garbage collection
            import gc

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_initialized = False
            logger.info("Classifier model resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up classifier model: {str(e)}")


# Load configuration
config = load_config()

# Create model instances
primary_model = PrimaryModel(config["primary_model"])
embedding_model = EmbeddingModel(config["embedding_model"])
classifier_model = ClassifierModel(config["classifier_model"])


async def initialize_models():
    """Initialize all models."""
    tasks = []

    # Start with embedding model as it's typically the smallest
    tasks.append(embedding_model.initialize())

    # Then initialize primary and classifier models
    tasks.append(primary_model.initialize())
    tasks.append(classifier_model.initialize())

    # Wait for all models to initialize
    await asyncio.gather(*tasks)
    logger.info("All models initialized")


async def cleanup_models():
    """Clean up all model resources."""
    tasks = []

    tasks.append(primary_model.cleanup())
    tasks.append(embedding_model.cleanup())
    tasks.append(classifier_model.cleanup())

    # Wait for all cleanup tasks to complete
    await asyncio.gather(*tasks)
    logger.info("All model resources cleaned up")
