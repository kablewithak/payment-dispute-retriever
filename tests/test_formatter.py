from __future__ import annotations

from app.services.formatter import RetrievalFormatter
from app.schemas.retrieval import (
    QueryFilters,
    RetrievedCase,
    RetrieveRequest,
    RetrievalMode,
)


def build_case(
    *,
    rank: int,
    issue_family: str,
    escalation_team: str,
    score: float | None,
    evidence: list[str],
    rerank_score: float | None = None,
) -> RetrievedCase:
    return RetrievedCase(
        rank=rank,
        dispute_id=f"D-{rank:05d}",
        case_title=f"Case {rank}",
        issue_family=issue_family,
        scheme="visa",
        payment_rail="card",
        region="uk",
        merchant_category="saas",
        amount_bucket="100_to_500",
        outcome="merchant_prevailed",
        escalation_team=escalation_team,
        resolution_summary="Resolution summary",
        evidence_submitted=evidence,
        score=score,
        distance=None,
        rerank_score=rerank_score,
        explain_score="hybrid explanation",
        match_summary="scheme matched; payment_rail matched",
    )


def test_formatter_selects_weighted_issue_family_and_evidence() -> None:
    formatter = RetrievalFormatter()
    request = RetrieveRequest(
        query_text="duplicate charge after retry timeout",
        mode=RetrievalMode.HYBRID_FILTERED_RERANK,
        filters=QueryFilters(scheme="visa", payment_rail="card"),
    )
    candidates = [
        build_case(
            rank=1,
            issue_family="duplicate_capture_after_retry",
            escalation_team="payments_platform_disputes",
            score=0.92,
            rerank_score=3.2,
            evidence=["gateway retry trace", "idempotency key inspection"],
        ),
        build_case(
            rank=2,
            issue_family="duplicate_capture_after_retry",
            escalation_team="payments_platform_disputes",
            score=0.88,
            rerank_score=2.1,
            evidence=["gateway retry trace", "webhook delivery log"],
        ),
        build_case(
            rank=3,
            issue_family="delayed_presentment_confusion",
            escalation_team="disputes_ops",
            score=0.41,
            rerank_score=0.4,
            evidence=["authorization-presentment timeline"],
        ),
    ]

    response = formatter.format(
        query_id="QRY-test",
        request=request,
        candidates=candidates,
    )

    assert response.predicted_issue_family == "duplicate_capture_after_retry"
    assert response.predicted_escalation_team == "payments_platform_disputes"
    assert "gateway retry trace" in response.recommended_evidence
    assert response.confidence > 0.0
    assert response.recommended_action is not None


def test_formatter_handles_no_results() -> None:
    formatter = RetrievalFormatter()
    request = RetrieveRequest(
        query_text="unknown query",
        mode=RetrievalMode.BM25,
    )

    response = formatter.format(
        query_id="QRY-empty",
        request=request,
        candidates=[],
    )

    assert response.retrieved_count == 0
    assert response.confidence == 0.0
    assert response.top_cases == []