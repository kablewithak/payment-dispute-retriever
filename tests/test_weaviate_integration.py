from __future__ import annotations

import os

import pytest

from app.services.weaviate_client import weaviate_client_context
from app.settings import get_settings


@pytest.mark.skipif(
    os.getenv("RUN_WEAVIATE_INTEGRATION") != "1",
    reason="Set RUN_WEAVIATE_INTEGRATION=1 to run the live Weaviate integration test.",
)
def test_local_weaviate_is_ready() -> None:
    settings = get_settings()
    with weaviate_client_context(settings) as client:
        assert client.is_ready()