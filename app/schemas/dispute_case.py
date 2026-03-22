from __future__ import annotations

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PaymentRail(str, Enum):
    CARD = "card"
    WALLET = "wallet"
    BANK_TRANSFER = "bank_transfer"


class Scheme(str, Enum):
    VISA = "visa"
    MASTERCARD = "mastercard"
    AMEX = "amex"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    SEPA = "sepa"


class Region(str, Enum):
    UK = "uk"
    EU = "eu"
    US = "us"
    SOUTH_AFRICA = "south_africa"
    UAE = "uae"


class MerchantCategory(str, Enum):
    DIGITAL_GOODS = "digital_goods"
    TRAVEL = "travel"
    SAAS = "saas"
    SUBSCRIPTION_SERVICES = "subscription_services"
    RETAIL = "retail"
    LOGISTICS = "logistics"
    FOOD_DELIVERY = "food_delivery"
    GAMING = "gaming"


class MerchantSize(str, Enum):
    STARTUP = "startup"
    MID_MARKET = "mid_market"
    ENTERPRISE = "enterprise"


class AmountBucket(str, Enum):
    UNDER_25 = "under_25"
    BETWEEN_25_AND_100 = "25_to_100"
    BETWEEN_100_AND_500 = "100_to_500"
    BETWEEN_500_AND_2000 = "500_to_2000"
    ABOVE_2000 = "above_2000"


class Outcome(str, Enum):
    MERCHANT_PREVAILED = "merchant_prevailed"
    CARDHOLDER_PREVAILED = "cardholder_prevailed"
    REPRESENTMENT_REQUIRED = "representment_required"
    GOODWILL_REFUND = "goodwill_refund"
    WRITE_OFF = "write_off"


class EscalationTeam(str, Enum):
    DISPUTES_OPS = "disputes_ops"
    FRAUD_OPS = "fraud_ops"
    MERCHANT_INTEGRATIONS = "merchant_integrations"
    PAYMENTS_PLATFORM_DISPUTES = "payments_platform_disputes"
    CHARGEBACK_QA = "chargeback_qa"


class IssueFamily(str, Enum):
    DUPLICATE_CAPTURE_AFTER_RETRY = "duplicate_capture_after_retry"
    THREE_DS_AUTHENTICATION_TIMEOUT = "3ds_authentication_timeout"
    FRAUD_CARD_NOT_PRESENT = "fraud_card_not_present"
    FRIENDLY_FRAUD_DIGITAL_GOODS = "friendly_fraud_digital_goods"
    SUBSCRIPTION_CANCELLATION_CLAIM = "subscription_cancellation_claim"
    REFUND_NOT_PROCESSED_CLAIM = "refund_not_processed_claim"
    CHARGEBACK_MISSING_DESCRIPTOR = "chargeback_missing_descriptor"
    MERCHANT_DESCRIPTOR_MISMATCH = "merchant_descriptor_mismatch"
    DELAYED_PRESENTMENT_CONFUSION = "delayed_presentment_confusion"
    PARTIAL_SHIPMENT_CLAIM = "partial_shipment_claim"


class DisputeCase(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    dispute_id: str = Field(..., description="Stable unique dispute case identifier.")
    case_title: str = Field(..., min_length=10, max_length=200)
    case_summary: str = Field(..., min_length=40)
    customer_claim: str = Field(..., min_length=20)
    merchant_response_summary: str = Field(..., min_length=20)
    processor_notes: str = Field(..., min_length=20)
    evidence_submitted: list[str] = Field(..., min_length=1)
    resolution_summary: str = Field(..., min_length=20)

    issue_family: IssueFamily
    reason_code: str = Field(..., description="Synthetic but realistic payment dispute reason code.")
    scheme: Scheme
    payment_rail: PaymentRail
    region: Region

    merchant_name: str = Field(..., min_length=3, max_length=100)
    merchant_category: MerchantCategory
    merchant_size: MerchantSize
    amount_bucket: AmountBucket
    currency: str = Field(..., min_length=3, max_length=3)

    risk_flags: list[str] = Field(..., min_length=1)
    outcome: Outcome
    escalation_team: EscalationTeam

    created_at: datetime

    @field_validator("dispute_id")
    @classmethod
    def validate_dispute_id(cls, value: str) -> str:
        if not re.fullmatch(r"D-\d{5}", value):
            raise ValueError("dispute_id must match D-00000 format")
        return value

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str) -> str:
        if not re.fullmatch(r"\d{2}(?:\.\d{1,2})?", value):
            raise ValueError("reason_code must look like 10.4 or 13.7")
        return value

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Z]{3}", value):
            raise ValueError("currency must be a 3-letter ISO-style code")
        return value

    @field_validator("evidence_submitted", "risk_flags")
    @classmethod
    def validate_non_empty_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("list must contain at least one non-empty item")
        return list(dict.fromkeys(cleaned))

    def to_retrieval_text(self) -> str:
        return " | ".join(
            [
                self.case_title,
                self.case_summary,
                self.customer_claim,
                self.merchant_response_summary,
                self.processor_notes,
                self.resolution_summary,
                self.issue_family.value,
                self.reason_code,
                self.scheme.value,
                self.payment_rail.value,
                self.region.value,
                self.merchant_category.value,
            ]
        )


class EvalQuery(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query_id: str = Field(..., description="Stable unique query identifier.")
    query_text: str = Field(..., min_length=15)
    filters: dict[str, str] = Field(default_factory=dict)
    gold_issue_family: IssueFamily
    gold_relevant_dispute_ids: list[str] = Field(..., min_length=1)
    gold_escalation_team: EscalationTeam

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, value: str) -> str:
        if not re.fullmatch(r"Q-\d{4}", value):
            raise ValueError("query_id must match Q-0000 format")
        return value

    @field_validator("gold_relevant_dispute_ids")
    @classmethod
    def validate_gold_ids(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("gold_relevant_dispute_ids must not be empty")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("gold_relevant_dispute_ids must be unique")
        return cleaned