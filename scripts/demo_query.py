from __future__ import annotations

import argparse

from app.schemas.retrieval import QueryFilters, RetrieveRequest, RetrievalMode
from app.services.embeddings import EmbeddingService
from app.services.formatter import RetrievalFormatter
from app.services.reranker import CrossEncoderReranker
from app.services.weaviate_client import weaviate_client_context
from app.services.workflow import execute_retrieval_request
from app.settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local demo retrieval query against Weaviate."
    )
    parser.add_argument("--query", required=True, help="Natural language dispute query.")
    parser.add_argument("--payment-rail", default=None, help="Optional payment rail filter.")
    parser.add_argument("--scheme", default=None, help="Optional scheme filter.")
    parser.add_argument("--region", default=None, help="Optional region filter.")
    parser.add_argument("--reason-code", default=None, help="Optional reason code filter.")
    parser.add_argument("--limit", type=int, default=5, help="Number of final results.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Hybrid alpha value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    filters = QueryFilters(
        payment_rail=args.payment_rail,
        scheme=args.scheme,
        region=args.region,
        reason_code=args.reason_code,
    )

    embedding_service = EmbeddingService(model_name=settings.embedding_model_name)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model_name)
    formatter = RetrievalFormatter()

    with weaviate_client_context(settings) as client:
        for mode in (
            RetrievalMode.BM25,
            RetrievalMode.VECTOR,
            RetrievalMode.HYBRID,
            RetrievalMode.HYBRID_FILTERED,
            RetrievalMode.HYBRID_FILTERED_RERANK,
        ):
            request = RetrieveRequest(
                query_text=args.query,
                mode=mode,
                limit=args.limit,
                alpha=args.alpha,
                candidate_pool_size=max(settings.rerank_candidate_pool_size, args.limit),
                filters=filters,
            )
            response = execute_retrieval_request(
                query_id=f"demo-{mode.value}",
                request=request,
                client=client,
                settings=settings,
                embedding_service=embedding_service,
                formatter=formatter,
                reranker=reranker,
            )
            print("=" * 100)
            print(mode.value.upper())
            print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()