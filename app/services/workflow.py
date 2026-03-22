from __future__ import annotations

from app.schemas.retrieval import RetrieveRequest, RetrieveResponse, RetrievalMode
from app.services.embeddings import EmbeddingService
from app.services.formatter import RetrievalFormatter
from app.services.reranker import CrossEncoderReranker
from app.services.retriever import DisputeCaseRetriever
from app.settings import Settings
from weaviate import WeaviateClient


def execute_retrieval_request(
    *,
    query_id: str,
    request: RetrieveRequest,
    client: WeaviateClient,
    settings: Settings,
    embedding_service: EmbeddingService,
    formatter: RetrievalFormatter,
    reranker: CrossEncoderReranker | None = None,
) -> RetrieveResponse:
    retriever = DisputeCaseRetriever(
        client=client,
        settings=settings,
        embedding_service=embedding_service,
    )

    if request.mode == RetrievalMode.HYBRID_FILTERED_RERANK:
        if reranker is None:
            raise RuntimeError("Reranker is required for hybrid_filtered_rerank mode")

        base_request = request.model_copy(
            update={
                "mode": RetrievalMode.HYBRID_FILTERED,
                "limit": request.candidate_pool_size,
            }
        )
        base_candidates = retriever.search(base_request)
        candidates = reranker.rerank(
            query_text=request.query_text,
            candidates=base_candidates,
            top_k=request.limit,
        )
    else:
        candidates = retriever.search(request)

    return formatter.format(
        query_id=query_id,
        request=request,
        candidates=candidates,
    )