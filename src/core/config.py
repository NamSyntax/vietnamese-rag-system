from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


# app settings class
class Settings(BaseSettings):
    # general api config
    PROJECT_NAME: str = "Vietnamese RAG System"
    API_V1_STR: str = "/api/v1"

    # qdrant vector db
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)

    # redis cache status
    REDIS_URL: str = Field(default="redis://localhost:6379")

    # --- LLM config ---
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434/api/chat")
    LLM_MODEL_NAME: str = Field(default="qwen2.5:7b-instruct")
    LLM_TEMPERATURE: float = Field(default=0.1)

    # bot persona
    BOT_NAME: str = "VietRAG Bot"
    CREATOR_NAME: str = "NamSyntax"

    # env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


# lru cache (singleton)
# cached settings instance
@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
