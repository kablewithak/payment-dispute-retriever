from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.logging import get_logger
from app.routes.retrieve import router as retrieve_router
from app.services.embeddings import EmbeddingService
from app.services.reranker import CrossEncoderReranker
from app.settings import get_settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_directories()

    embedding_service = EmbeddingService(model_name=settings.embedding_model_name)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model_name)

    app.state.settings = settings
    app.state.embedding_service = embedding_service
    app.state.reranker = reranker

    logger.info(
        "application_started",
        extra={
            "collection_name": settings.weaviate_collection_name,
            "embedding_model_name": settings.embedding_model_name,
            "reranker_model_name": settings.reranker_model_name,
        },
    )
    yield
    logger.info("application_stopped")


app = FastAPI(
    title="Payment Dispute Retriever",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(retrieve_router)