from __future__ import annotations

from typing import Any, Sequence

from sentence_transformers import CrossEncoder

from app.schemas.retrieval import RetrievedCase


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        model: Any | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model = model

    def rerank(
        self,
        *,
        query_text: str,
        candidates: Sequence[RetrievedCase],
        top_k: int | None = None,
    ) -> list[RetrievedCase]:
        if not candidates:
            return []

        pairs = [
            (query_text, self._build_candidate_text(candidate))
            for candidate in candidates
        ]
        scores = self._get_model().predict(pairs)

        reranked: list[RetrievedCase] = []
        for candidate, score in zip(candidates, scores, strict=True):
            reranked.append(
                candidate.model_copy(
                    update={"rerank_score": float(score)}
                )
            )

        reranked.sort(
            key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
            reverse=True,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        return [
            candidate.model_copy(update={"rank": index})
            for index, candidate in enumerate(reranked, start=1)
        ]

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )
        return self._model

    def _build_candidate_text(self, candidate: RetrievedCase) -> str:
        return " | ".join(
            [
                candidate.case_title,
                candidate.issue_family.value,
                candidate.scheme.value,
                candidate.payment_rail.value,
                candidate.region.value,
                candidate.merchant_category.value,
                candidate.resolution_summary,
                " ".join(candidate.evidence_submitted),
                candidate.match_summary,
            ]
        )