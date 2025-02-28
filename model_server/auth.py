"""
Enhanced authentication module for model server with rate limiting and expiration
"""

import os
import json
import secrets
import logging
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from fastapi import HTTPException, Header, Depends, Request
from fastapi.security import APIKeyHeader

logger = logging.getLogger("model-server")

# Constants
API_KEYS_FILE = os.environ.get("API_KEYS_FILE", "api_keys.json")
DEFAULT_RATE_LIMIT = 100  # Requests per minute
DEFAULT_KEY_EXPIRY_DAYS = 365  # 1 year


class RateLimiter:
    """Rate limiter implementation using token bucket algorithm"""

    def __init__(self):
        self.buckets = {}
        self.lock = asyncio.Lock()

    async def check_rate_limit(self, api_key: str, limit: Optional[int]) -> Tuple[bool, float]:
        """
        Check if request is within rate limit
        Returns: (allowed: bool, retry_after: float)
        """
        if limit is None or limit <= 0:
            return True, 0  # No rate limit

        async with self.lock:
            now = time.time()

            # Initialize bucket if not exists
            if api_key not in self.buckets:
                self.buckets[api_key] = {"tokens": limit, "last_refill": now, "capacity": limit}

            bucket = self.buckets[api_key]

            # Refill tokens based on time elapsed (1 minute refill period)
            time_passed = now - bucket["last_refill"]
            refill_amount = time_passed * (bucket["capacity"] / 60.0)

            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + refill_amount)
            bucket["last_refill"] = now

            # Check if we have enough tokens
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True, 0
            else:
                # Calculate time until next token available
                time_until_refill = (1 - bucket["tokens"]) / (bucket["capacity"] / 60.0)
                return False, time_until_refill


class APIKeyManager:
    def __init__(self, keys_file: str = API_KEYS_FILE):
        self.keys_file = keys_file
        self.api_keys: Dict[str, Dict[str, any]] = {}
        self.rate_limiter = RateLimiter()
        self._load_api_keys()

    def _load_api_keys(self) -> None:
        """Load API keys from file."""
        try:
            if os.path.exists(self.keys_file):
                with open(self.keys_file, "r") as f:
                    self.api_keys = json.load(f)

                # Migrate old keys to new format if needed
                updated = False
                for key, data in self.api_keys.items():
                    # Add expiry if not present
                    if "expires_at" not in data:
                        data["expires_at"] = (datetime.now() + timedelta(days=DEFAULT_KEY_EXPIRY_DAYS)).isoformat()
                        updated = True

                    # Ensure rate limit exists
                    if "rate_limit" not in data:
                        data["rate_limit"] = DEFAULT_RATE_LIMIT if data.get("role") != "admin" else None
                        updated = True

                if updated:
                    self._save_api_keys()

                # Count active keys
                active_keys = sum(1 for data in self.api_keys.values() if data.get("active", False))
                logger.info(f"Loaded {len(self.api_keys)} API keys ({active_keys} active)")
            else:
                # Create default admin key if file doesn't exist
                admin_key = self.generate_api_key()
                self.api_keys = {
                    admin_key: {
                        "name": "admin",
                        "role": "admin",
                        "active": True,
                        "rate_limit": None,  # No rate limit for admin
                        "created_at": datetime.now().isoformat(),
                        "expires_at": (datetime.now() + timedelta(days=DEFAULT_KEY_EXPIRY_DAYS)).isoformat(),
                        "last_used": None,
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
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.keys_file) or ".", exist_ok=True)

            with open(self.keys_file, "w") as f:
                json.dump(self.api_keys, f, indent=2)
            logger.info(f"Saved {len(self.api_keys)} API keys")
        except Exception as e:
            logger.error(f"Error saving API keys: {str(e)}")

    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return secrets.token_urlsafe(48)  # Longer key for better security

    def create_api_key(self, name: str, role: str = "client", rate_limit: Optional[int] = DEFAULT_RATE_LIMIT, expires_in_days: Optional[int] = DEFAULT_KEY_EXPIRY_DAYS) -> str:
        """Create a new API key and save it."""
        new_key = self.generate_api_key()

        # Admin keys have no rate limit by default
        if role == "admin" and rate_limit is None:
            rate_limit = None

        # Calculate expiration date
        expires_at = None
        if expires_in_days is not None:
            expires_at = (datetime.now() + timedelta(days=expires_in_days)).isoformat()

        self.api_keys[new_key] = {"name": name, "role": role, "active": True, "rate_limit": rate_limit, "created_at": datetime.now().isoformat(), "expires_at": expires_at, "last_used": None}

        self._save_api_keys()
        return new_key

    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key without checking rate limits."""
        key_data = self.api_keys.get(api_key)

        if not key_data:
            return False

        # Check if key is active
        if not key_data.get("active", False):
            return False

        # Check if key has expired
        expires_at = key_data.get("expires_at")
        if expires_at and datetime.now() > datetime.fromisoformat(expires_at):
            # Automatically deactivate expired keys
            key_data["active"] = False
            self._save_api_keys()
            return False

        # Update last used timestamp
        key_data["last_used"] = datetime.now().isoformat()

        return True

    async def check_api_key(self, api_key: str) -> Tuple[bool, Optional[float], Optional[Dict]]:
        """
        Validate an API key and check rate limits.
        Returns: (allowed: bool, retry_after: Optional[float], key_data: Optional[Dict])
        """
        if not self.validate_api_key(api_key):
            return False, None, None

        key_data = self.api_keys.get(api_key)

        # Check rate limit
        allowed, retry_after = await self.rate_limiter.check_rate_limit(api_key, key_data.get("rate_limit"))

        if not allowed:
            return False, retry_after, key_data

        return True, None, key_data

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
            # Don't allow changing certain fields
            safe_data = {k: v for k, v in key_data.items() if k not in ["created_at", "last_used"]}

            self.api_keys[api_key].update(safe_data)
            self._save_api_keys()
            return True
        return False

    def get_all_keys(self) -> Dict[str, Dict[str, any]]:
        """Get all API keys."""
        return self.api_keys

    def get_key_info(self, api_key: str) -> Optional[Dict[str, any]]:
        """Get information about a specific API key."""
        return self.api_keys.get(api_key)


# Create global instance
api_key_manager = APIKeyManager()

# Create API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def validate_api_key(request: Request, api_key: str = Depends(api_key_header)) -> str:
    """Validate API key from header with rate limiting."""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is missing",
            headers={"WWW-Authenticate": "APIKey"},
        )

    allowed, retry_after, key_data = await api_key_manager.check_api_key(api_key)

    if not allowed:
        if retry_after:
            # Rate limit exceeded
            headers = {"Retry-After": str(int(retry_after))}
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {int(retry_after)} seconds.",
                headers=headers,
            )
        else:
            # Invalid key
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired API key",
                headers={"WWW-Authenticate": "APIKey"},
            )

    # Add key info to request state for logging/metrics
    request.state.api_key_info = {"name": key_data.get("name"), "role": key_data.get("role")}

    return api_key


async def validate_admin_api_key(request: Request, api_key: str = Depends(api_key_header)) -> str:
    """Validate that the API key belongs to an admin."""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is missing",
            headers={"WWW-Authenticate": "APIKey"},
        )

    allowed, retry_after, key_data = await api_key_manager.check_api_key(api_key)

    if not allowed:
        if retry_after:
            headers = {"Retry-After": str(int(retry_after))}
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {int(retry_after)} seconds.",
                headers=headers,
            )
        else:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired API key",
                headers={"WWW-Authenticate": "APIKey"},
            )

    # Check if key has admin role
    if key_data.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required for this operation",
        )

    # Add key info to request state
    request.state.api_key_info = {"name": key_data.get("name"), "role": key_data.get("role")}

    return api_key


def generate_new_key(name: str, role: str = "client", rate_limit: Optional[int] = DEFAULT_RATE_LIMIT, expires_in_days: Optional[int] = DEFAULT_KEY_EXPIRY_DAYS) -> str:
    """Generate a new API key with expiration and rate limiting."""
    return api_key_manager.create_api_key(name=name, role=role, rate_limit=rate_limit, expires_in_days=expires_in_days)

