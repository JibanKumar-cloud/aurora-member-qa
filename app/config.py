from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl

BASE_DIR = Path(__file__).resolve().parent.parent  # /app/app -> /app
DATA_DIR = BASE_DIR / "data"

class Settings(BaseSettings):
    # This will be overridden by .env if present
    messages_api_url: AnyHttpUrl = "http://localhost:9000/messages"

    top_k: int = 3

    # FAISS / persistence config
    faiss_index_path: str = str(DATA_DIR / "faiss.index")
    faiss_meta_path: str = str(DATA_DIR / "messages_meta.json")


    # Index type: "flat" or "ivfpq"
    faiss_index_type: str = "ivfpq"

    # IVF-PQ params (used if faiss_index_type == "ivfpq")
    faiss_nlist: int = 4096      # number of clusters
    faiss_m: int = 64            # number of subvectors for PQ
    faiss_nprobe: int = 16       # how many clusters to search at query time

    class Config:
        env_file = ".env"


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
