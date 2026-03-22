from __future__ import annotations

from app.logging import get_logger
from app.schemas.retrieval import RetrievalMode
from app.services.embeddings import EmbeddingService
from app.services.evaluator import RetrievalEvaluator
from app.services.formatter import RetrievalFormatter
from app.services.reranker import CrossEncoderReranker
from app.services.weaviate_client import weaviate_client_context
from app.services.workflow import execute_retrieval_request
from app.settings import get_settings

logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    settings.ensure_directories()

    evaluator = RetrievalEvaluator(
        rerank_candidate_pool_size=settings.rerank_candidate_pool_size
    )
    eval_queries = evaluator.load_eval_queries(
        settings.eval_dir / "eval_queries.jsonl"
    )

    embedding_service = EmbeddingService(model_name=settings.embedding_model_name)
    reranker = CrossEncoderReranker(model_name=settings.reranker_model_name)
    formatter = RetrievalFormatter()

    summaries = []
    all_rows = []

    with weaviate_client_context(settings) as client:
        def run_request(query_id: str, request):
            return execute_retrieval_request(
                query_id=query_id,
                request=request,
                client=client,
                settings=settings,
                embedding_service=embedding_service,
                formatter=formatter,
                reranker=reranker,
            )

        for mode in (
            RetrievalMode.BM25,
            RetrievalMode.VECTOR,
            RetrievalMode.HYBRID,
            RetrievalMode.HYBRID_FILTERED,
            RetrievalMode.HYBRID_FILTERED_RERANK,
        ):
            summary, rows = evaluator.evaluate_mode(
                eval_queries=eval_queries,
                mode=mode,
                run_request=run_request,
            )
            summaries.append(summary)
            all_rows.extend(rows)
            logger.info(
                "evaluation_mode_completed",
                extra={
                    "mode": summary.mode,
                    "query_count": summary.query_count,
                    "recall_at_5": summary.recall_at_5,
                    "mrr_at_10": summary.mrr_at_10,
                    "hit_rate_at_3": summary.hit_rate_at_3,
                    "issue_family_accuracy": summary.issue_family_accuracy,
                    "escalation_team_accuracy": summary.escalation_team_accuracy,
                    "avg_latency_ms": summary.avg_latency_ms,
                },
            )

    evaluator.write_summary_json(
        summaries=summaries,
        output_path=settings.metrics_dir / "summary.json",
    )
    evaluator.write_summary_csv(
        summaries=summaries,
        output_path=settings.metrics_dir / "summary.csv",
    )
    evaluator.write_query_rows_csv(
        rows=all_rows,
        output_path=settings.metrics_dir / "query_rows.csv",
    )

    logger.info(
        "evaluation_artifacts_written",
        extra={
            "summary_json": str(settings.metrics_dir / "summary.json"),
            "summary_csv": str(settings.metrics_dir / "summary.csv"),
            "query_rows_csv": str(settings.metrics_dir / "query_rows.csv"),
        },
    )


if __name__ == "__main__":
    main()