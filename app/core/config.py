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
        # OpenAI Configuration
        self.openai_api_key = self._get_required("OPENAI_API_KEY")
        
        # Database Configuration
        self.vector_db_url = self._get_required("VECTOR_DB_CONNECTION_STRING")
        self.state_db_url = self._get_required("CONVERSATION_DB_CONNECTION_STRING")
        
        # Linode Object Storage Configuration
        self.linode_object_storage_access_key = self._get_required("S3_ACCESS_KEY")
        self.linode_object_storage_secret_key = self._get_required("S3_SECRET_KEY")
        self.linode_object_storage_endpoint = self._ensure_https_prefix(
            self._get_required("S3_ENDPOINT")
        )
        self.linode_object_storage_bucket = self._get_required("S3_BUCKET_NAME")
        
        # Application Configuration
        self.app_host = os.getenv("APP_HOST", "0.0.0.0")
        self.app_port = int(os.getenv("APP_PORT", "8000"))
        
        # Logging Configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # RAG Configuration
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "10"))
        
        # OpenAI Model Configuration
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Admin API Configuration
        # ADMIN_API_KEY should contain the actual API key value
        self.admin_api_key = self._get_required("ADMIN_API_KEY")
    
    def _get_required(self, env_var: str) -> str:
        """Get a required environment variable, raising an error if missing."""
        value = os.getenv(env_var)
        if not value:
            raise ValueError(f"Required environment variable {env_var} is not set")
        return value
    
    def _ensure_https_prefix(self, endpoint: str) -> str:
        """Ensure S3 endpoint has https:// prefix."""
        if endpoint and not endpoint.startswith(('http://', 'https://')):
            return f"https://{endpoint}"
        return endpoint


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
        "OPENAI_API_KEY",
        "VECTOR_DB_CONNECTION_STRING", 
        "CONVERSATION_DB_CONNECTION_STRING",
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
        "S3_ENDPOINT",
        "S3_BUCKET_NAME",
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
            # Log first few chars for sensitive values, full value for non-sensitive
            if "KEY" in var or "SECRET" in var or "PASSWORD" in var or "CONNECTION_STRING" in var:
                masked = value[:20] + "..." if len(value) > 20 else "***"
                logger.debug(f"✓ {var}: {masked} (length: {len(value)})")
            else:
                logger.debug(f"✓ {var}: {value}")
        else:
            missing_vars.append(var)
            logger.warning(f"✗ {var}: NOT SET")
    
    logger.info(f"Found {len(found_vars)}/{len(required_vars)} required environment variables")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("All required environment variables are set")
    return True
