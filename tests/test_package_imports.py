from __future__ import annotations

import importlib


def test_synthetic_data_module_imports_cleanly() -> None:
    module = importlib.import_module("app.services.synthetic_data")
    assert hasattr(module, "SyntheticDisputeDataGenerator")


def test_retriever_module_exposes_dispute_case_retriever() -> None:
    module = importlib.import_module("app.services.retriever")
    assert hasattr(module, "DisputeCaseRetriever")