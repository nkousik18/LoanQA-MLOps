# app/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ------------------------------
    # vLLM SETTINGS
    # ------------------------------
    VLLM_API_URL: str = "http://127.0.0.1:9000"
    VLLM_MODEL: str = "mistral7b"

    # ------------------------------
    # EMBEDDINGS / RETRIEVER
    # ------------------------------
    EMBEDDER_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTORSTORE_PATH: str = "vectorstore/index"
    VECTORSTORE_COLLECTION: str = "loan_docs_collection"
    RETRIEVER_TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 7000

    # ------------------------------
    # SECURITY
    # ------------------------------
    API_KEY: str
    HMAC_SECRET: str

    # ------------------------------
    # APP CONFIG
    # ------------------------------
    APP_NAME: str = "loan-doc-ai-inference"
    LOG_LEVEL: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

