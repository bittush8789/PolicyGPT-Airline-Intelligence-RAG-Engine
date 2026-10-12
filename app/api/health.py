"""
Health check API endpoints for monitoring application status.
Provides system health information and connectivity status.
"""

import logging
from typing import Dict, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime

from app.models.schemas import HealthResponse
from app.core.config import get_settings
from app.core.rag import get_rag_pipeline
from app.core.memory import get_conversation_memory

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


async def check_vector_db() -> str:
    """Check vector database connectivity."""
    try:
        rag_pipeline = get_rag_pipeline()
        stats = rag_pipeline.get_vector_store_stats()
        return "connected" if stats["status"] == "connected" else "error"
    except Exception as e:
        logger.error(f"Vector database check failed: {e}")
        return "error"


async def check_state_db() -> str:
    """Check state database connectivity."""
    try:
        conversation_memory = get_conversation_memory()
        stats = conversation_memory.get_checkpointer_stats()
        return "connected" if stats["status"] == "connected" else "error"
    except Exception as e:
        logger.error(f"State database check failed: {e}")
        return "error"


async def check_groq_api() -> str:
    """Check Groq API availability."""
    try:
        if not settings.groq_api_key or settings.groq_api_key.startswith("gsk_your"):
            return "not_configured"
            
        from langchain_groq import ChatGroq
        
        # Test with a simple lightweight query
        llm = ChatGroq(
            model=settings.llm_model,
            groq_api_key=settings.groq_api_key,
            temperature=0,
            max_tokens=5
        )
        
        response = llm.invoke("Hi")
        if response and response.content:
            return "available"
        else:
            return "error"
            
    except Exception as e:
        logger.error(f"Groq API check failed: {e}")
        return "unavailable"


# Legacy alias
async def check_openai_api() -> str:
    return await check_groq_api()


@router.get("/health")
async def health_check(
    services: Optional[bool] = Query(False, description="If true, check all external services")
) -> Union[Dict[str, str], HealthResponse]:
    """
    Perform health check of the application.
    
    By default, returns a simple {"status": "healthy"} response without checking external services.
    Use ?services=true to check vector database, state database, and Groq API.
    
    Args:
        services: If true, perform checks on all external services
        
    Returns:
        Simple dict with status, or HealthResponse with full service status
    """
    try:
        if not services:
            # Simple health check - just return healthy without checking services
            logger.info("Performing simple health check")
            return {"status": "healthy"}
        
        # Full health check with service checks
        logger.info("Performing comprehensive health check with service checks")
        
        # Check all components
        vector_db_status = await check_vector_db()
        state_db_status = await check_state_db()
        groq_status = await check_groq_api()
        
        # Determine overall status
        overall_status = "healthy"
        if vector_db_status == "error" or state_db_status == "error":
            overall_status = "unhealthy"
        elif groq_status == "unavailable":
            overall_status = "degraded"
        
        response = HealthResponse(
            status=overall_status,
            vector_db=vector_db_status,
            state_db=state_db_status,
            groq_api=groq_status,
            openai_api=groq_status
        )
        
        logger.info(f"Health check completed: {overall_status}")
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        if not services:
            return {"status": "error"}
        return HealthResponse(
            status="error",
            vector_db="unknown",
            state_db="unknown",
            groq_api="unknown",
            openai_api="unknown"
        )


@router.get("/health/vector-db")
async def check_vector_database() -> Dict[str, Any]:
    """
    Check vector database connectivity and status.
    
    Returns:
        Dictionary with vector database status information
    """
    try:
        rag_pipeline = get_rag_pipeline()
        stats = rag_pipeline.get_vector_store_stats()
        
        logger.info("Vector database check completed")
        return stats
        
    except Exception as e:
        logger.error(f"Vector database check failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/health/state-db")
async def check_state_database() -> Dict[str, Any]:
    """
    Check state database connectivity and status.
    
    Returns:
        Dictionary with state database status information
    """
    try:
        conversation_memory = get_conversation_memory()
        stats = conversation_memory.get_checkpointer_stats()
        
        logger.info("State database check completed")
        return stats
        
    except Exception as e:
        logger.error(f"State database check failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/health/groq")
async def check_groq() -> Dict[str, Any]:
    """
    Check Groq API connectivity and status.
    
    Returns:
        Dictionary with Groq API status information
    """
    try:
        status = await check_groq_api()
        
        result = {
            "status": status,
            "model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info("Groq API check completed")
        return result
        
    except Exception as e:
        logger.error(f"Groq API check failed: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/health/openai")
async def check_openai() -> Dict[str, Any]:
    """Legacy alias for check_groq."""
    return await check_groq()


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Perform detailed health check with component-specific information.
    
    Returns:
        Dictionary with detailed health information
    """
    try:
        logger.info("Performing detailed health check")
        
        # Check all components
        vector_db_status = await check_vector_db()
        state_db_status = await check_state_db()
        groq_status = await check_groq_api()
        
        # Get detailed stats
        rag_pipeline = get_rag_pipeline()
        conversation_memory = get_conversation_memory()
        
        vector_stats = rag_pipeline.get_vector_store_stats()
        memory_stats = conversation_memory.get_checkpointer_stats()
        
        # Determine overall status
        overall_status = "healthy"
        if vector_db_status == "error" or state_db_status == "error":
            overall_status = "unhealthy"
        elif groq_status == "unavailable":
            overall_status = "degraded"
        
        result = {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "vector_database": {
                    "status": vector_db_status,
                    "details": vector_stats
                },
                "state_database": {
                    "status": state_db_status,
                    "details": memory_stats
                },
                "groq_api": {
                    "status": groq_status,
                    "model": settings.llm_model,
                    "embedding_model": settings.embedding_model
                }
            },
            "configuration": {
                "app_host": settings.app_host,
                "app_port": settings.app_port,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "retrieval_k": settings.retrieval_k
            }
        }
        
        logger.info(f"Detailed health check completed: {overall_status}")
        return result
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "overall_status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
