from __future__ import annotations

from app.schemas.retrieval import RetrievedCase
from app.services.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs):
        return [0.2, 1.4, 0.6]


def build_case(rank: int, dispute_id: str) -> RetrievedCase:
    return RetrievedCase(
        rank=rank,
        dispute_id=dispute_id,
        case_title=f"Case {rank}",
        issue_family="duplicate_capture_after_retry",
        scheme="visa",
        payment_rail="card",
        region="uk",
        merchant_category="saas",
        amount_bucket="100_to_500",
        outcome="merchant_prevailed",
        escalation_team="payments_platform_disputes",
        resolution_summary="Resolution summary",
        evidence_submitted=["gateway retry trace"],
        score=0.5,
        match_summary="initial retrieval hit",
    )


def test_reranker_reorders_candidates_by_cross_encoder_score() -> None:
    reranker = CrossEncoderReranker(
        model_name="fake-model",
        model=FakeCrossEncoder(),
    )
    candidates = [
        build_case(1, "D-00001"),
        build_case(2, "D-00002"),
        build_case(3, "D-00003"),
    ]

    reranked = reranker.rerank(
        query_text="duplicate charge after retry timeout",
        candidates=candidates,
        top_k=2,
    )

    assert len(reranked) == 2
    assert reranked[0].dispute_id == "D-00002"
    assert reranked[0].rerank_score == 1.4
    assert reranked[0].rank == 1
    assert reranked[1].rank == 2