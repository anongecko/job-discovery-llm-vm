"""
Enhanced Model Server for AI Job Discovery System
"""

__version__ = "2.0.0"

# Core components for easier imports
from model_server.main import app
from model_server.auth import generate_new_key, api_key_manager, validate_api_key, validate_admin_api_key
from model_server.models import primary_model, embedding_model, classifier_model, initialize_models, cleanup_models
from model_server.api_keys import router as api_keys_router

# Version info
VERSION_INFO = {
    "version": __version__,
    "models": {
        "primary": {"name": "llama-3.1-70b-instruct", "type": "gguf"},
        "embedding": {"name": "nomic-embed-text-v1", "type": "huggingface"},
        "classifier": {"name": "mistral-7b-instruct-v0.2", "type": "gguf"},
    },
}
