from __future__ import annotations

from app.services.indexer import (
    build_weaviate_properties,
    build_weaviate_uuid,
    prepare_index_records,
)
from app.services.synthetic_data import SyntheticDisputeDataGenerator


class FakeEmbeddingService:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), float(index + 1), float(index + 1)] for index, _ in enumerate(texts)]


def test_build_weaviate_uuid_is_stable() -> None:
    generator = SyntheticDisputeDataGenerator(seed=42)
    cases = generator.generate_cases(case_count=20)

    first_uuid = build_weaviate_uuid(cases[0])
    second_uuid = build_weaviate_uuid(cases[0])
    different_uuid = build_weaviate_uuid(cases[1])

    assert first_uuid == second_uuid
    assert first_uuid != different_uuid


def test_build_weaviate_properties_contains_retrieval_text() -> None:
    generator = SyntheticDisputeDataGenerator(seed=42)
    case = generator.generate_cases(case_count=20)[0]

    properties = build_weaviate_properties(case)

    assert properties["dispute_id"] == case.dispute_id
    assert properties["issue_family"] == case.issue_family.value
    assert properties["retrieval_text"]
    assert isinstance(properties["evidence_submitted"], list)
    assert isinstance(properties["risk_flags"], list)


def test_prepare_index_records_uses_named_default_vector() -> None:
    generator = SyntheticDisputeDataGenerator(seed=42)
    cases = generator.generate_cases(case_count=20)[:3]

    records = prepare_index_records(
        cases=cases,
        embedding_service=FakeEmbeddingService(),
    )

    assert len(records) == 3
    assert set(records[0].vector.keys()) == {"default"}
    assert records[0].properties["dispute_id"] == cases[0].dispute_id
    assert len(records[0].vector["default"]) == 3