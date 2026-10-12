"""
Configuration management for the LangChain RAG Chatbot application.
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory (parent of app/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env file into environment variables
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        """Initialize settings from environment variables."""
        # Groq Configuration
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        
        # Database Configuration (PostgreSQL with pgvector & LangGraph checkpoints)
        self.vector_db_url = self._get_required("VECTOR_DB_CONNECTION_STRING")
        self.state_db_url = self._get_required("CONVERSATION_DB_CONNECTION_STRING")
        
        # Local Documents Directory
        self.documents_dir = os.getenv("DOCUMENTS_DIR", str(PROJECT_ROOT / "documents"))
        
        # Application Configuration
        self.app_host = os.getenv("APP_HOST", "0.0.0.0")
        self.app_port = int(os.getenv("APP_PORT", "8000"))
        
        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # RAG Configuration
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "10"))
        
        # Groq Model Configuration
        self.llm_model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        
        # Admin API Configuration
        self.admin_api_key = self._get_required("ADMIN_API_KEY")
        
        # LangSmith Monitoring Configuration
        self.langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")
        self.langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
        self.langchain_project = os.getenv("LANGCHAIN_PROJECT", "policygpt-airline-rag")
        self.langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    
    def _get_required(self, env_var: str) -> str:
        """Get a required environment variable, raising an error if missing."""
        value = os.getenv(env_var)
        if not value:
            raise ValueError(f"Required environment variable {env_var} is not set")
        return value


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings instance."""
    return settings


def validate_environment() -> bool:
    """Validate that all required environment variables are set."""
    import logging
    logger = logging.getLogger(__name__)
    
    required_vars = [
        "VECTOR_DB_CONNECTION_STRING", 
        "CONVERSATION_DB_CONNECTION_STRING",
        "ADMIN_API_KEY"
    ]
    
    logger.info("Validating environment variables...")
    logger.debug(f"Checking for {len(required_vars)} required environment variables")
    
    missing_vars = []
    found_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            found_vars.append(var)
            if "KEY" in var or "SECRET" in var or "PASSWORD" in var or "CONNECTION_STRING" in var:
                masked = value[:20] + "..." if len(value) > 20 else "***"
                logger.debug(f"✓ {var}: {masked} (length: {len(value)})")
            else:
                logger.debug(f"✓ {var}: {value}")
        else:
            missing_vars.append(var)
            logger.warning(f"✗ {var}: NOT SET")
            
    # Check GROQ_API_KEY
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key.startswith("gsk_your"):
        logger.warning("! GROQ_API_KEY is not configured or is a placeholder. Chat responses will require a valid Groq API key.")
    else:
        logger.info(f"✓ GROQ_API_KEY: {groq_key[:8]}... (configured)")
        found_vars.append("GROQ_API_KEY")
    
    logger.info(f"Found {len(found_vars)} required/configured environment variables")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    # Configure and validate LangSmith Tracing
    if settings.langchain_tracing_v2 and settings.langchain_api_key and not settings.langchain_api_key.startswith("lsv2_pt_your"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        logger.info(f"✓ LangSmith Tracing V2: ENABLED (Project: '{settings.langchain_project}')")
    else:
        logger.info("ℹ LangSmith Tracing V2: Disabled (set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to monitor in LangSmith)")

    logger.info("All required database environment variables are set")
    return True
