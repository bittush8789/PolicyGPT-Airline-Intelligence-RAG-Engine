"""
Pydantic models for API request and response schemas.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="The user's message", min_length=1, max_length=4000)
    thread_id: Optional[str] = Field(None, description="Conversation thread ID for context")
    airline: Optional[str] = Field(None, description="Optional airline name to filter policies")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="The assistant's response")
    thread_id: str = Field(..., description="The conversation thread ID")
    sources: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Retrieved sources")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class AirlinesResponse(BaseModel):
    """Response model for list of available airlines."""
    airlines: List[str] = Field(..., description="List of unique indexed airlines")
    total_chunks: int = Field(0, description="Total chunks indexed across all airlines")



class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Overall application status")
    vector_db: str = Field(..., description="Vector database connection status")
    state_db: str = Field(..., description="State database connection status")
    groq_api: str = Field(..., description="Groq API availability status")
    openai_api: Optional[str] = Field(None, description="Legacy field for backwards compatibility")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")


