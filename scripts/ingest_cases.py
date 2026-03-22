from __future__ import annotations

import argparse
from pathlib import Path

from app.logging import get_logger
from app.services.embeddings import EmbeddingService
from app.services.indexer import (
    WeaviateDisputeIndexer,
    load_cases_from_jsonl,
    prepare_index_records,
)
from app.services.weaviate_client import weaviate_client_context
from app.settings import get_settings


def parse_args() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Embed and ingest synthetic dispute cases into local Weaviate."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=settings.synthetic_dir / "dispute_cases.jsonl",
        help="Path to the synthetic dispute JSONL file.",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Recreate the collection before ingestion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    logger = get_logger(__name__)

    cases = load_cases_from_jsonl(args.input_path)
    logger.info(
        "loaded_cases_from_disk",
        extra={"input_path": args.input_path, "case_count": len(cases)},
    )

    embedding_service = EmbeddingService(model_name=settings.embedding_model_name)
    records = prepare_index_records(cases=cases, embedding_service=embedding_service)

    with weaviate_client_context(settings) as client:
        indexer = WeaviateDisputeIndexer(client=client, settings=settings)

        if args.recreate_collection:
            indexer.recreate_collection()
        else:
            indexer.ensure_collection()

        result = indexer.index_records(
            records=records,
            batch_size=settings.weaviate_batch_size,
        )

    logger.info(
        "weaviate_ingestion_completed",
        extra={
            "collection_name": settings.weaviate_collection_name,
            "attempted_count": result["attempted_count"],
            "inserted_count": result["inserted_count"],
            "failed_count": result["failed_count"],
        },
    )


if __name__ == "__main__":
    main()