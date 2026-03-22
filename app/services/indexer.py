from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from weaviate import WeaviateClient
from weaviate.classes.config import Configure, DataType, Property, VectorDistances

from app.schemas.dispute_case import DisputeCase
from app.services.embeddings import EmbeddingService
from app.settings import Settings


@dataclass(frozen=True)
class WeaviateObjectRecord:
    uuid: str
    properties: dict[str, Any]
    vector: dict[str, list[float]]


def build_weaviate_uuid(case: DisputeCase) -> str:
    return str(uuid5(NAMESPACE_URL, f"payment-dispute-retriever:{case.dispute_id}"))


def build_weaviate_properties(case: DisputeCase) -> dict[str, Any]:
    return {
        "dispute_id": case.dispute_id,
        "case_title": case.case_title,
        "case_summary": case.case_summary,
        "customer_claim": case.customer_claim,
        "merchant_response_summary": case.merchant_response_summary,
        "processor_notes": case.processor_notes,
        "evidence_submitted": case.evidence_submitted,
        "resolution_summary": case.resolution_summary,
        "issue_family": case.issue_family.value,
        "reason_code": case.reason_code,
        "scheme": case.scheme.value,
        "payment_rail": case.payment_rail.value,
        "region": case.region.value,
        "merchant_name": case.merchant_name,
        "merchant_category": case.merchant_category.value,
        "merchant_size": case.merchant_size.value,
        "amount_bucket": case.amount_bucket.value,
        "currency": case.currency,
        "risk_flags": case.risk_flags,
        "outcome": case.outcome.value,
        "escalation_team": case.escalation_team.value,
        "created_at": case.created_at.isoformat(),
        "retrieval_text": case.to_retrieval_text(),
    }


def prepare_index_records(
    cases: list[DisputeCase],
    embedding_service: EmbeddingService,
) -> list[WeaviateObjectRecord]:
    retrieval_texts = [case.to_retrieval_text() for case in cases]
    vectors = embedding_service.embed_texts(retrieval_texts)

    return [
        WeaviateObjectRecord(
            uuid=build_weaviate_uuid(case),
            properties=build_weaviate_properties(case),
            vector={"default": vector},
        )
        for case, vector in zip(cases, vectors, strict=True)
    ]


def load_cases_from_jsonl(path: Path) -> list[DisputeCase]:
    cases: list[DisputeCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            cases.append(DisputeCase.model_validate_json(stripped))
    return cases


class WeaviateDisputeIndexer:
    def __init__(self, client: WeaviateClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def ensure_collection(self) -> bool:
        if self.client.collections.exists(self.settings.weaviate_collection_name):
            return False
        self._create_collection()
        return True

    def recreate_collection(self) -> None:
        if self.client.collections.exists(self.settings.weaviate_collection_name):
            self.client.collections.delete(self.settings.weaviate_collection_name)
        self._create_collection()

    def _create_collection(self) -> None:
        self.client.collections.create(
            name=self.settings.weaviate_collection_name,
            vector_config=Configure.Vectors.self_provided(
                name="default",
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                ),
            ),
            properties=[
                Property(name="dispute_id", data_type=DataType.TEXT),
                Property(name="case_title", data_type=DataType.TEXT),
                Property(name="case_summary", data_type=DataType.TEXT),
                Property(name="customer_claim", data_type=DataType.TEXT),
                Property(name="merchant_response_summary", data_type=DataType.TEXT),
                Property(name="processor_notes", data_type=DataType.TEXT),
                Property(name="evidence_submitted", data_type=DataType.TEXT_ARRAY),
                Property(name="resolution_summary", data_type=DataType.TEXT),
                Property(name="issue_family", data_type=DataType.TEXT),
                Property(name="reason_code", data_type=DataType.TEXT),
                Property(name="scheme", data_type=DataType.TEXT),
                Property(name="payment_rail", data_type=DataType.TEXT),
                Property(name="region", data_type=DataType.TEXT),
                Property(name="merchant_name", data_type=DataType.TEXT),
                Property(name="merchant_category", data_type=DataType.TEXT),
                Property(name="merchant_size", data_type=DataType.TEXT),
                Property(name="amount_bucket", data_type=DataType.TEXT),
                Property(name="currency", data_type=DataType.TEXT),
                Property(name="risk_flags", data_type=DataType.TEXT_ARRAY),
                Property(name="outcome", data_type=DataType.TEXT),
                Property(name="escalation_team", data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.DATE),
                Property(name="retrieval_text", data_type=DataType.TEXT),
            ],
        )

    def index_records(
        self,
        records: Iterable[WeaviateObjectRecord],
        *,
        batch_size: int,
    ) -> dict[str, int]:
        collection = self.client.collections.use(self.settings.weaviate_collection_name)
        records_list = list(records)

        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for record in records_list:
                batch.add_object(
                    properties=record.properties,
                    uuid=record.uuid,
                    vector=record.vector,
                )
                if batch.number_errors > 10:
                    break

        failed_objects = collection.batch.failed_objects
        failed_count = len(failed_objects)
        inserted_count = len(records_list) - failed_count

        return {
            "attempted_count": len(records_list),
            "inserted_count": inserted_count,
            "failed_count": failed_count,
        }

    def export_failed_objects(self) -> str:
        collection = self.client.collections.use(self.settings.weaviate_collection_name)
        return json.dumps([str(item) for item in collection.batch.failed_objects], indent=2)