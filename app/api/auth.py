"""
API key authentication for admin endpoints.
"""

import logging
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# API Key header
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(x_api_key: str = Security(api_key_header)) -> str:
    """
    Verify the API key from the x-api-key header.
    
    The stored ADMIN_API_KEY environment variable should contain the actual API key.
    
    Args:
        x_api_key: The API key from the request header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If the API key is missing or invalid
    """
    if not x_api_key:
        logger.warning("API key missing from request")
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Please provide x-api-key header."
        )
    
    stored_key = settings.admin_api_key
    
    if x_api_key != stored_key:
        logger.warning("Invalid API key attempted")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key."
        )
    
    logger.debug("API key validated successfully")
    return x_api_key

