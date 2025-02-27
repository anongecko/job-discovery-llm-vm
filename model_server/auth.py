"""
Authentication module for model server
"""

import os
import json
import secrets
import logging
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import HTTPException, Header, Depends

logger = logging.getLogger("model-server")

# Constants
API_KEYS_FILE = "api_keys.json"


class APIKeyManager:
    def __init__(self, keys_file: str = API_KEYS_FILE):
        self.keys_file = keys_file
        self.api_keys: Dict[str, Dict[str, any]] = {}
        self._load_api_keys()

    def _load_api_keys(self) -> None:
        """Load API keys from file."""
        try:
            if os.path.exists(self.keys_file):
                with open(self.keys_file, "r") as f:
                    self.api_keys = json.load(f)
                logger.info(f"Loaded {len(self.api_keys)} API keys")
            else:
                # Create default admin key if file doesn't exist
                admin_key = self.generate_api_key()
                self.api_keys = {
                    admin_key: {
                        "name": "admin",
                        "role": "admin",
                        "active": True,
                        "rate_limit": None,  # No rate limit
                    }
                }
                self._save_api_keys()
                logger.info(f"Created default admin API key: {admin_key}")
        except Exception as e:
            logger.error(f"Error loading API keys: {str(e)}")
            # Create empty keys dict to avoid errors
            self.api_keys = {}

    def _save_api_keys(self) -> None:
        """Save API keys to file."""
        try:
            with open(self.keys_file, "w") as f:
                json.dump(self.api_keys, f, indent=2)
            logger.info(f"Saved {len(self.api_keys)} API keys")
        except Exception as e:
            logger.error(f"Error saving API keys: {str(e)}")

    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return secrets.token_urlsafe(32)

    def create_api_key(self, name: str, role: str = "client", rate_limit: Optional[int] = 100) -> str:
        """Create a new API key and save it."""
        new_key = self.generate_api_key()
        self.api_keys[new_key] = {"name": name, "role": role, "active": True, "rate_limit": rate_limit}
        self._save_api_keys()
        return new_key

    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key."""
        key_data = self.api_keys.get(api_key)
        if key_data and key_data.get("active", False):
            return True
        return False

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            self._save_api_keys()
            return True
        return False

    def update_api_key(self, api_key: str, key_data: Dict[str, any]) -> bool:
        """Update API key data."""
        if api_key in self.api_keys:
            self.api_keys[api_key].update(key_data)
            self._save_api_keys()
            return True
        return False

    def get_all_keys(self) -> Dict[str, Dict[str, any]]:
        """Get all API keys."""
        return self.api_keys


# Create global instance
api_key_manager = APIKeyManager()


async def validate_api_key(api_key: str = Header(None, alias="X-API-Key")) -> str:
    """Validate API key from header."""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is missing")

    if not api_key_manager.validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key


def generate_new_key(name: str, role: str = "client") -> str:
    """Generate a new API key."""
    return api_key_manager.create_api_key(name, role)
