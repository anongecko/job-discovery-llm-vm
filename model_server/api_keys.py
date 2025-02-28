"""
API Key management endpoints for the model server
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends, Request
from model_server.auth import api_key_manager, validate_admin_api_key, generate_new_key, DEFAULT_RATE_LIMIT, DEFAULT_KEY_EXPIRY_DAYS

logger = logging.getLogger("model-server")

# Create API router
router = APIRouter(tags=["API Keys"], prefix="/api-keys")


class APIKeyResponse(BaseModel):
    key: str = Field(..., description="API key")
    name: str = Field(..., description="Name of the key owner")
    role: str = Field(..., description="Role of the key owner")
    active: bool = Field(..., description="Whether the key is active")
    rate_limit: Optional[int] = Field(None, description="Rate limit in requests per minute")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
    last_used: Optional[str] = Field(None, description="Last used timestamp")


class APIKeyInfo(BaseModel):
    name: str = Field(..., description="Name of the key owner")
    role: str = Field(..., description="Role of the key owner")
    rate_limit: Optional[int] = Field(DEFAULT_RATE_LIMIT, description="Rate limit in requests per minute")
    expires_in_days: Optional[int] = Field(DEFAULT_KEY_EXPIRY_DAYS, description="Key validity in days")


class APIKeyUpdateInfo(BaseModel):
    name: Optional[str] = Field(None, description="Name of the key owner")
    role: Optional[str] = Field(None, description="Role of the key owner")
    active: Optional[bool] = Field(None, description="Whether the key is active")
    rate_limit: Optional[int] = Field(None, description="Rate limit in requests per minute")
    expires_in_days: Optional[int] = Field(None, description="Key validity in days from now")


@router.post("/create", response_model=APIKeyResponse)
async def create_api_key(key_info: APIKeyInfo, api_key: str = Depends(validate_admin_api_key)):
    """Create a new API key (admin only)."""
    try:
        new_key = generate_new_key(name=key_info.name, role=key_info.role, rate_limit=key_info.rate_limit, expires_in_days=key_info.expires_in_days)

        key_data = api_key_manager.get_key_info(new_key)

        return {"key": new_key, **key_data}
    except Exception as e:
        logger.error(f"Error creating API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create API key: {str(e)}")


@router.get("/list", response_model=List[APIKeyResponse])
async def list_api_keys(api_key: str = Depends(validate_admin_api_key), include_inactive: bool = False):
    """List all API keys (admin only)."""
    try:
        all_keys = api_key_manager.get_all_keys()

        # Filter keys based on active status if needed
        if not include_inactive:
            all_keys = {k: v for k, v in all_keys.items() if v.get("active", False)}

        # Format response
        result = [{"key": k, **v} for k, v in all_keys.items()]

        return result
    except Exception as e:
        logger.error(f"Error listing API keys: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list API keys: {str(e)}")


@router.get("/info", response_model=APIKeyResponse)
async def get_api_key_info(request: Request, api_key: str = Depends(validate_admin_api_key), key_to_check: Optional[str] = None):
    """
    Get information about an API key (admin only).
    If key_to_check is not specified, returns info about the current key.
    """
    try:
        target_key = key_to_check if key_to_check else api_key

        key_data = api_key_manager.get_key_info(target_key)
        if not key_data:
            raise HTTPException(status_code=404, detail=f"API key not found")

        return {"key": target_key, **key_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting API key info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get API key info: {str(e)}")


@router.patch("/update/{target_key}", response_model=APIKeyResponse)
async def update_api_key(target_key: str, update_info: APIKeyUpdateInfo, api_key: str = Depends(validate_admin_api_key)):
    """Update an API key (admin only)."""
    try:
        key_data = api_key_manager.get_key_info(target_key)
        if not key_data:
            raise HTTPException(status_code=404, detail=f"API key not found")

        # Prepare update data
        update_data = {k: v for k, v in update_info.dict().items() if v is not None}

        # Handle expiration update specially
        if update_info.expires_in_days is not None:
            update_data["expires_at"] = (datetime.now() + timedelta(days=update_info.expires_in_days)).isoformat()
            del update_data["expires_in_days"]

        # Update key
        success = api_key_manager.update_api_key(target_key, update_data)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to update API key")

        # Get updated info
        updated_key_data = api_key_manager.get_key_info(target_key)

        return {"key": target_key, **updated_key_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update API key: {str(e)}")


@router.delete("/revoke/{target_key}")
async def revoke_api_key(target_key: str, api_key: str = Depends(validate_admin_api_key)):
    """Revoke an API key (admin only)."""
    try:
        # Prevent revoking the current admin key
        if target_key == api_key:
            raise HTTPException(status_code=400, detail="Cannot revoke the current admin key")

        success = api_key_manager.revoke_api_key(target_key)
        if not success:
            raise HTTPException(status_code=404, detail=f"API key not found or already revoked")

        return {"message": "API key revoked successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to revoke API key: {str(e)}")


@router.post("/regenerate")
async def regenerate_api_key(api_key_info: APIKeyInfo, old_key: str, api_key: str = Depends(validate_admin_api_key)):
    """Regenerate an API key (admin only)."""
    try:
        # Check if old key exists
        old_key_data = api_key_manager.get_key_info(old_key)
        if not old_key_data:
            raise HTTPException(status_code=404, detail=f"API key not found")

        # Create new key
        new_key = generate_new_key(name=api_key_info.name, role=api_key_info.role, rate_limit=api_key_info.rate_limit, expires_in_days=api_key_info.expires_in_days)

        # Revoke old key
        api_key_manager.revoke_api_key(old_key)

        # Get new key data
        new_key_data = api_key_manager.get_key_info(new_key)

        return {"key": new_key, **new_key_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error regenerating API key: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to regenerate API key: {str(e)}")
