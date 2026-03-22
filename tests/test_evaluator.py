from __future__ import annotations

from app.schemas.dispute_case import EvalQuery
from app.schemas.retrieval import RetrieveResponse, RetrievedCase, RetrievalMode
from app.services.evaluator import RetrievalEvaluator


def build_response(
    *,
    query_id: str,
    predicted_issue_family: str,
    predicted_escalation_team: str,
    retrieved_ids: list[str],
) -> RetrieveResponse:
    return RetrieveResponse(
        query_id=query_id,
        mode="hybrid_filtered_rerank",
        query_text="query",
        filters_applied={},
        retrieved_count=len(retrieved_ids),
        predicted_issue_family=predicted_issue_family,
        predicted_escalation_team=predicted_escalation_team,
        confidence=0.82,
        recommended_action="action",
        recommended_evidence=["evidence"],
        top_cases=[
            RetrievedCase(
                rank=index,
                dispute_id=dispute_id,
                case_title=f"Case {index}",
                issue_family=predicted_issue_family,
                scheme="visa",
                payment_rail="card",
                region="uk",
                merchant_category="saas",
                amount_bucket="100_to_500",
                outcome="merchant_prevailed",
                escalation_team=predicted_escalation_team,
                resolution_summary="Resolution summary",
                evidence_submitted=["gateway retry trace"],
                match_summary="match",
            )
            for index, dispute_id in enumerate(retrieved_ids, start=1)
        ],
    )


def test_evaluator_computes_mode_summary() -> None:
    evaluator = RetrievalEvaluator(rerank_candidate_pool_size=15)

    eval_queries = [
        EvalQuery(
            query_id="Q-0001",
            query_text="duplicate charge after retry",
            filters={"payment_rail": "card", "scheme": "visa", "region": "uk"},
            gold_issue_family="duplicate_capture_after_retry",
            gold_relevant_dispute_ids=["D-00001", "D-00002"],
            gold_escalation_team="payments_platform_disputes",
        ),
        EvalQuery(
            query_id="Q-0002",
            query_text="refund initiated but not visible",
            filters={"payment_rail": "card", "scheme": "visa", "region": "uk"},
            gold_issue_family="refund_not_processed_claim",
            gold_relevant_dispute_ids=["D-00010"],
            gold_escalation_team="disputes_ops",
        ),
    ]

    response_map = {
        "Q-0001": build_response(
            query_id="Q-0001",
            predicted_issue_family="duplicate_capture_after_retry",
            predicted_escalation_team="payments_platform_disputes",
            retrieved_ids=["D-00002", "D-00050", "D-00001"],
        ),
        "Q-0002": build_response(
            query_id="Q-0002",
            predicted_issue_family="refund_not_processed_claim",
            predicted_escalation_team="disputes_ops",
            retrieved_ids=["D-00111", "D-00010"],
        ),
    }

    def run_request(query_id, request):
        return response_map[query_id]

    summary, rows = evaluator.evaluate_mode(
        eval_queries=eval_queries,
        mode=RetrievalMode.HYBRID_FILTERED_RERANK,
        run_request=run_request,
    )

    assert len(rows) == 2
    assert summary.mode == "hybrid_filtered_rerank"
    assert summary.query_count == 2
    assert summary.recall_at_5 > 0.0
    assert summary.mrr_at_10 > 0.0
    assert summary.hit_rate_at_3 == 1.0
    assert summary.issue_family_accuracy == 1.0
    assert summary.escalation_team_accuracy == 1.0