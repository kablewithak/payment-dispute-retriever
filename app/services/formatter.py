from __future__ import annotations

import math
from collections import Counter, defaultdict

from app.schemas.dispute_case import EscalationTeam, IssueFamily
from app.schemas.retrieval import RetrieveRequest, RetrieveResponse, RetrievedCase


ISSUE_ACTION_MAP: dict[IssueFamily, str] = {
    IssueFamily.DUPLICATE_CAPTURE_AFTER_RETRY: "Review retry orchestration, idempotency enforcement, and capture chronology before liability response.",
    IssueFamily.THREE_DS_AUTHENTICATION_TIMEOUT: "Inspect 3DS challenge timing, ACS logs, and redirect completion before classifying as fraud or customer confusion.",
    IssueFamily.FRAUD_CARD_NOT_PRESENT: "Escalate to fraud review with device, AVS/CVV, and prior customer linkage evidence.",
    IssueFamily.FRIENDLY_FRAUD_DIGITAL_GOODS: "Prioritize entitlement, usage, and login telemetry to challenge the non-receipt narrative.",
    IssueFamily.SUBSCRIPTION_CANCELLATION_CLAIM: "Audit cancellation timing, renewal cutoffs, and customer notification evidence.",
    IssueFamily.REFUND_NOT_PROCESSED_CLAIM: "Verify refund initiation, reference IDs, and issuer posting windows before conceding the dispute.",
    IssueFamily.CHARGEBACK_MISSING_DESCRIPTOR: "Confirm statement descriptor mapping and checkout brand-to-entity alignment.",
    IssueFamily.MERCHANT_DESCRIPTOR_MISMATCH: "Check parent-brand aliasing and descriptor configuration for avoidable recognition disputes.",
    IssueFamily.DELAYED_PRESENTMENT_CONFUSION: "Reconstruct authorization, fulfillment, and presentment chronology to rule out duplicate-charge assumptions.",
    IssueFamily.PARTIAL_SHIPMENT_CLAIM: "Gather split-shipment, fulfillment, and customer support evidence before responding on charge validity.",
}


class RetrievalFormatter:
    def format(
        self,
        *,
        query_id: str,
        request: RetrieveRequest,
        candidates: list[RetrievedCase],
    ) -> RetrieveResponse:
        if not candidates:
            return RetrieveResponse(
                query_id=query_id,
                mode=request.mode,
                query_text=request.query_text,
                filters_applied=request.filters.active_filters(),
                retrieved_count=0,
                confidence=0.0,
                recommended_action="No matches returned. Relax filters or adjust the query wording.",
                recommended_evidence=[],
                top_cases=[],
            )

        predicted_issue_family = self._pick_issue_family(candidates)
        predicted_escalation_team = self._pick_escalation_team(candidates)
        recommended_evidence = self._pick_evidence(candidates)
        confidence = self._estimate_confidence(candidates)

        return RetrieveResponse(
            query_id=query_id,
            mode=request.mode,
            query_text=request.query_text,
            filters_applied=request.filters.active_filters(),
            retrieved_count=len(candidates),
            predicted_issue_family=predicted_issue_family,
            predicted_escalation_team=predicted_escalation_team,
            confidence=confidence,
            recommended_action=ISSUE_ACTION_MAP.get(predicted_issue_family),
            recommended_evidence=recommended_evidence,
            top_cases=candidates,
        )

    def _pick_issue_family(self, candidates: list[RetrievedCase]) -> IssueFamily:
        weights: dict[IssueFamily, float] = defaultdict(float)
        for rank, candidate in enumerate(candidates[:5], start=1):
            weights[candidate.issue_family] += self._candidate_weight(candidate, rank)
        return max(weights, key=weights.get)

    def _pick_escalation_team(
        self,
        candidates: list[RetrievedCase],
    ) -> EscalationTeam:
        weights: dict[EscalationTeam, float] = defaultdict(float)
        for rank, candidate in enumerate(candidates[:5], start=1):
            weights[candidate.escalation_team] += self._candidate_weight(candidate, rank)
        return max(weights, key=weights.get)

    def _pick_evidence(self, candidates: list[RetrievedCase]) -> list[str]:
        counter: Counter[str] = Counter()
        for candidate in candidates[:3]:
            counter.update(candidate.evidence_submitted)
        return [item for item, _ in counter.most_common(4)]

    def _estimate_confidence(self, candidates: list[RetrievedCase]) -> float:
        top_candidates = candidates[:3]
        family_counter = Counter(candidate.issue_family for candidate in top_candidates)
        consensus = max(family_counter.values()) / len(top_candidates)

        margin_signal = self._margin_signal(candidates)
        confidence = 0.35 + (0.4 * consensus) + (0.25 * margin_signal)
        return round(max(0.05, min(confidence, 0.95)), 4)

    def _margin_signal(self, candidates: list[RetrievedCase]) -> float:
        first = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        if first.rerank_score is not None:
            first_value = first.rerank_score
            second_value = second.rerank_score if second and second.rerank_score is not None else first_value - 0.5
            return 1 / (1 + math.exp(-(first_value - second_value)))

        if first.distance is not None:
            second_value = second.distance if second and second.distance is not None else min(first.distance + 0.15, 1.0)
            margin = max(second_value - first.distance, 0.0)
            return min(margin / 0.5, 1.0)

        if first.score is not None:
            second_value = second.score if second and second.score is not None else 0.0
            margin = max(first.score - second_value, 0.0)
            return margin / (margin + 1.0)

        return 0.5

    def _candidate_weight(self, candidate: RetrievedCase, rank: int) -> float:
        rank_weight = 1.0 / rank

        if candidate.rerank_score is not None:
            normalized = 1 / (1 + math.exp(-candidate.rerank_score))
            return rank_weight * normalized

        if candidate.score is not None:
            normalized = candidate.score / (abs(candidate.score) + 1.0)
            return rank_weight * max(normalized, 0.0001)

        if candidate.distance is not None:
            return rank_weight * (1.0 / (1.0 + max(candidate.distance, 0.0)))

        return rank_weight