from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    project_name: str = "payment-dispute-retriever"

    synthetic_case_count: int = Field(default=400, alias="SYNTHETIC_CASE_COUNT")
    eval_query_count: int = Field(default=50, alias="EVAL_QUERY_COUNT")
    random_seed: int = Field(default=42, alias="RANDOM_SEED")

    weaviate_host: str = Field(default="localhost", alias="WEAVIATE_HOST")
    weaviate_http_port: int = Field(default=8080, alias="WEAVIATE_HTTP_PORT")
    weaviate_grpc_port: int = Field(default=50051, alias="WEAVIATE_GRPC_PORT")
    weaviate_collection_name: str = Field(
        default="PaymentDisputeCase",
        alias="WEAVIATE_COLLECTION_NAME",
    )
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )
    weaviate_batch_size: int = Field(default=64, alias="WEAVIATE_BATCH_SIZE")

    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL_NAME",
    )
    rerank_candidate_pool_size: int = Field(
        default=15,
        alias="RERANK_CANDIDATE_POOL_SIZE",
    )

    data_dir: Path = REPO_ROOT / "data"
    synthetic_dir: Path = REPO_ROOT / "data" / "synthetic"
    eval_dir: Path = REPO_ROOT / "data" / "eval"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    metrics_dir: Path = REPO_ROOT / "artifacts" / "metrics"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()