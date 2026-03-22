from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.dispute_case import (
    AmountBucket,
    EscalationTeam,
    IssueFamily,
    MerchantCategory,
    PaymentRail,
    Region,
    Scheme,
)


class RetrievalMode(str, Enum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_FILTERED = "hybrid_filtered"
    HYBRID_FILTERED_RERANK = "hybrid_filtered_rerank"


class QueryFilters(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    payment_rail: PaymentRail | None = None
    scheme: Scheme | None = None
    region: Region | None = None
    reason_code: str | None = None
    merchant_category: MerchantCategory | None = None
    amount_bucket: AmountBucket | None = None

    def active_filters(self) -> dict[str, str]:
        active: dict[str, str] = {}
        for field_name, value in self.model_dump(exclude_none=True).items():
            active[field_name] = str(value)
        return active


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query_text: str = Field(..., min_length=10, max_length=1000)
    mode: RetrievalMode = RetrievalMode.HYBRID_FILTERED
    limit: int = Field(default=5, ge=1, le=10)
    alpha: float = Field(default=0.35, ge=0.0, le=1.0)
    candidate_pool_size: int = Field(default=15, ge=5, le=50)
    filters: QueryFilters = Field(default_factory=QueryFilters)

    @model_validator(mode="after")
    def validate_pool_size(self) -> "RetrieveRequest":
        if self.candidate_pool_size < self.limit:
            raise ValueError("candidate_pool_size must be greater than or equal to limit")
        return self


class RetrievedCase(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    rank: int
    dispute_id: str
    case_title: str
    issue_family: IssueFamily
    scheme: Scheme
    payment_rail: PaymentRail
    region: Region
    merchant_category: MerchantCategory
    amount_bucket: AmountBucket
    outcome: str
    escalation_team: EscalationTeam
    resolution_summary: str
    evidence_submitted: list[str]
    score: float | None = None
    distance: float | None = None
    rerank_score: float | None = None
    explain_score: str | None = None
    match_summary: str


class RetrieveResponse(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query_id: str
    mode: RetrievalMode
    query_text: str
    filters_applied: dict[str, str]
    retrieved_count: int
    predicted_issue_family: IssueFamily | None = None
    predicted_escalation_team: EscalationTeam | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recommended_action: str | None = None
    recommended_evidence: list[str] = Field(default_factory=list)
    top_cases: list[RetrievedCase] = Field(default_factory=list)