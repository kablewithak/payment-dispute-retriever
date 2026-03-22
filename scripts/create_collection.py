from __future__ import annotations

import argparse

from app.logging import get_logger
from app.services.indexer import WeaviateDisputeIndexer
from app.services.weaviate_client import weaviate_client_context
from app.settings import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or recreate the local Weaviate collection."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the collection first if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    logger = get_logger(__name__)

    with weaviate_client_context(settings) as client:
        indexer = WeaviateDisputeIndexer(client=client, settings=settings)

        if args.recreate:
            indexer.recreate_collection()
            logger.info(
                "weaviate_collection_recreated",
                extra={"collection_name": settings.weaviate_collection_name},
            )
            return

        created = indexer.ensure_collection()
        logger.info(
            "weaviate_collection_checked",
            extra={
                "collection_name": settings.weaviate_collection_name,
                "created": created,
            },
        )


if __name__ == "__main__":
    main()