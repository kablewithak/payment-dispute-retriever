from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import weaviate
from weaviate import WeaviateClient

from app.settings import Settings


class WeaviateConnectionError(RuntimeError):
    """Raised when the Weaviate client cannot connect to a live local instance."""


def create_local_weaviate_client(settings: Settings) -> WeaviateClient:
    client = weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_http_port,
        grpc_port=settings.weaviate_grpc_port,
    )
    if not client.is_ready():
        client.close()
        raise WeaviateConnectionError(
            "Weaviate is not ready. Start Docker and verify ports 8080/50051."
        )
    return client


@contextmanager
def weaviate_client_context(settings: Settings) -> Iterator[WeaviateClient]:
    client = create_local_weaviate_client(settings)
    try:
        yield client
    finally:
        client.close()