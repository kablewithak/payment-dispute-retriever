from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Request

from app.logging import get_logger
from app.schemas.retrieval import RetrieveRequest, RetrieveResponse
from app.services.formatter import RetrievalFormatter
from app.services.weaviate_client import weaviate_client_context
from app.services.workflow import execute_retrieval_request
from app.settings import Settings

router = APIRouter()
logger = get_logger(__name__)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_disputes(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
    settings: Settings = request.app.state.settings
    embedding_service = request.app.state.embedding_service
    reranker = request.app.state.reranker

    started = perf_counter()
    query_id = f"QRY-{uuid4().hex[:12]}"

    with weaviate_client_context(settings) as client:
        response = execute_retrieval_request(
            query_id=query_id,
            request=payload,
            client=client,
            settings=settings,
            embedding_service=embedding_service,
            formatter=RetrievalFormatter(),
            reranker=reranker,
        )

    duration_ms = round((perf_counter() - started) * 1000, 2)
    logger.info(
        "retrieval_completed",
        extra={
            "query_id": query_id,
            "mode": payload.mode.value,
            "limit": payload.limit,
            "alpha": payload.alpha,
            "candidate_pool_size": payload.candidate_pool_size,
            "filters_applied": payload.filters.active_filters(),
            "retrieved_count": response.retrieved_count,
            "duration_ms": duration_ms,
        },
    )
    return response


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}