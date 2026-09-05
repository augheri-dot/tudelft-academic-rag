from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    # Azure Storage Configuration
    AZURE_STORAGE_CONNECTION_STRING: str = "dummy_connection_string"
    AZURE_CONTAINER_NAME: str = "tudelft-lakehouse"
    
    # OpenAI API Configuration
    OPENAI_API_KEY: str = "sk-dummy_api_key"
    
    # Vector Database & Caching Services
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    REDIS_HOST: str = "redis-cache"
    REDIS_PORT: int = 6379
    
    # Target Academic Institution (TU Delft Active OpenAlex ID: I98358874)
    TUDELFT_INSTITUTION_ID: str = "I98358874"
    OPENALEX_BASE_URL: str = "https://api.openalex.org"
    USER_EMAIL: str = "agustinus.heriyanto@maranatha.edu"
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
