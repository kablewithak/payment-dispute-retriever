from __future__ import annotations

from typing import Any

from weaviate import WeaviateClient
from weaviate.classes.query import Filter, MetadataQuery

from app.schemas.retrieval import (
    QueryFilters,
    RetrieveRequest,
    RetrievedCase,
    RetrievalMode,
)
from app.services.embeddings import EmbeddingService
from app.settings import Settings


class DisputeCaseRetriever:
    KEYWORD_QUERY_PROPERTIES = [
        "retrieval_text^4",
        "case_title^2",
        "customer_claim",
        "merchant_response_summary",
        "resolution_summary",
        "reason_code^3",
        "issue_family^2",
    ]

    def __init__(
        self,
        client: WeaviateClient,
        settings: Settings,
        embedding_service: EmbeddingService,
    ) -> None:
        self.client = client
        self.settings = settings
        self.embedding_service = embedding_service

    def search(self, request: RetrieveRequest) -> list[RetrievedCase]:
        collection = self.client.collections.use(self.settings.weaviate_collection_name)

        if request.mode == RetrievalMode.BM25:
            response = collection.query.bm25(
                query=request.query_text,
                query_properties=self.KEYWORD_QUERY_PROPERTIES,
                filters=self._build_weaviate_filter(request.filters),
                return_metadata=MetadataQuery(score=True, explain_score=True),
                limit=request.limit,
            )
        elif request.mode == RetrievalMode.VECTOR:
            query_vector = self.embedding_service.embed_text(request.query_text)
            response = collection.query.near_vector(
                near_vector=query_vector,
                target_vector="default",
                filters=self._build_weaviate_filter(request.filters),
                return_metadata=MetadataQuery(distance=True),
                limit=request.limit,
            )
        elif request.mode == RetrievalMode.HYBRID:
            query_vector = self.embedding_service.embed_text(request.query_text)
            response = collection.query.hybrid(
                query=request.query_text,
                vector=query_vector,
                target_vector="default",
                alpha=request.alpha,
                query_properties=self.KEYWORD_QUERY_PROPERTIES,
                return_metadata=MetadataQuery(score=True, explain_score=True),
                limit=request.limit,
            )
        elif request.mode == RetrievalMode.HYBRID_FILTERED:
            query_vector = self.embedding_service.embed_text(request.query_text)
            response = collection.query.hybrid(
                query=request.query_text,
                vector=query_vector,
                target_vector="default",
                alpha=request.alpha,
                query_properties=self.KEYWORD_QUERY_PROPERTIES,
                filters=self._build_weaviate_filter(request.filters),
                return_metadata=MetadataQuery(score=True, explain_score=True),
                limit=request.limit,
            )
        else:
            raise ValueError(f"Unsupported retrieval mode: {request.mode}")

        return [
            self._to_retrieved_case(
                rank=index,
                obj=obj,
                query_filters=request.filters,
            )
            for index, obj in enumerate(response.objects, start=1)
        ]

    def _build_weaviate_filter(self, filters: QueryFilters) -> Filter | None:
        clauses: list[Filter] = []

        if filters.payment_rail is not None:
            clauses.append(
                Filter.by_property("payment_rail").equal(filters.payment_rail.value)
            )
        if filters.scheme is not None:
            clauses.append(Filter.by_property("scheme").equal(filters.scheme.value))
        if filters.region is not None:
            clauses.append(Filter.by_property("region").equal(filters.region.value))
        if filters.reason_code is not None:
            clauses.append(Filter.by_property("reason_code").equal(filters.reason_code))
        if filters.merchant_category is not None:
            clauses.append(
                Filter.by_property("merchant_category").equal(
                    filters.merchant_category.value
                )
            )
        if filters.amount_bucket is not None:
            clauses.append(
                Filter.by_property("amount_bucket").equal(filters.amount_bucket.value)
            )

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return Filter.all_of(clauses)

    def _to_retrieved_case(
        self,
        *,
        rank: int,
        obj: Any,
        query_filters: QueryFilters,
    ) -> RetrievedCase:
        props = dict(obj.properties)
        metadata = getattr(obj, "metadata", None)

        score = getattr(metadata, "score", None)
        distance = getattr(metadata, "distance", None)
        explain_score = getattr(metadata, "explain_score", None)

        return RetrievedCase(
            rank=rank,
            dispute_id=props["dispute_id"],
            case_title=props["case_title"],
            issue_family=props["issue_family"],
            scheme=props["scheme"],
            payment_rail=props["payment_rail"],
            region=props["region"],
            merchant_category=props["merchant_category"],
            amount_bucket=props["amount_bucket"],
            outcome=props["outcome"],
            escalation_team=props["escalation_team"],
            resolution_summary=props["resolution_summary"],
            evidence_submitted=list(props.get("evidence_submitted", [])),
            score=float(score) if score is not None else None,
            distance=float(distance) if distance is not None else None,
            explain_score=str(explain_score) if explain_score is not None else None,
            match_summary=self._build_match_summary(
                props=props,
                query_filters=query_filters,
                score=float(score) if score is not None else None,
                distance=float(distance) if distance is not None else None,
            ),
        )

    def _build_match_summary(
        self,
        *,
        props: dict[str, Any],
        query_filters: QueryFilters,
        score: float | None,
        distance: float | None,
    ) -> str:
        reasons: list[str] = []

        active_filter_map = query_filters.active_filters()
        for field_name, expected_value in active_filter_map.items():
            actual_value = str(props.get(field_name))
            if actual_value == expected_value:
                reasons.append(f"{field_name} matched")

        if score is not None:
            reasons.append(f"hybrid/BM25 score={score:.4f}")
        if distance is not None:
            reasons.append(f"vector distance={distance:.4f}")

        if not reasons:
            reasons.append("semantic similarity from retrieval_text")
        return "; ".join(reasons)