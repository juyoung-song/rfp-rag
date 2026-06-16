from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    openai_api_key: str
    cohere_api_key: str = ""

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_name: str = "rfp_documents"

    data_dir: Path = Path("../data")
    top_k: int = 5
    active_phase: int = 1
    embedding_model: str = "openai"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
