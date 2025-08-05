import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Configurações globais do projeto Contextual AI.
    """

    API_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # Configurações de embeddings
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"

    # Configurações de LLM
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "changeme")

    # Banco vetorial (FAISS, Pinecone, Weaviate)
    VECTOR_DB_PROVIDER: str = "FAISS"
    VECTOR_DB_PATH: str = "./vector_store"

    class Config:
        env_file = ".env"

settings = Settings()
