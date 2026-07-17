"""
Central configuration for the RAG platform.

Everything that could plausibly change between environments (model names,
chunk sizes, file paths, API keys) lives here and nowhere else, loaded from
.env via pydantic-settings. No module in this project should read
os.environ directly - they should import `settings` from here instead.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gemini (embeddings)
    gemini_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-001"

    # Groq (generation)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Storage
    sqlite_db_path: str = "./rag_platform.db"
    chroma_persist_dir: str = "./chroma_data"
    upload_dir: str = "./uploads"

    # Chunking / retrieval
    pdf_chunk_token_size: int = 650
    pdf_chunk_overlap_tokens: int = 100
    default_top_k: int = 5

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def sqlalchemy_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.sqlite_db_path}"


settings = Settings()
