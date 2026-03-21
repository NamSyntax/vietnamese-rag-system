# src/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache

class Settings(BaseSettings):
    # --- Cấu hình API chung ---
    PROJECT_NAME: str = "Vietnamese RAG System"
    API_V1_STR: str = "/api/v1"
    
    # --- Qdrant (Vector DB) ---
    QDRANT_HOST: str = Field(default="localhost", env="QDRANT_HOST")
    QDRANT_PORT: int = Field(default=6333, env="QDRANT_PORT")
    
    # --- Redis (Cache/Status) ---
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # --- LLM config ---
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434/api/chat", env="OLLAMA_BASE_URL")
    LLM_MODEL_NAME: str = Field(default="qwen2.5:7b-instruct", env="LLM_MODEL_NAME")
    LLM_TEMPERATURE: float = Field(default=0.1, env="LLM_TEMPERATURE")
    
    # --- Persona Bot ---
    BOT_NAME: str = "VietRAG Bot"
    CREATOR_NAME: str = "NamSyntax"
    
    # file .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# lru cache (singleton)
@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()