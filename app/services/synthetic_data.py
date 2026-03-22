from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

from app.schemas.dispute_case import (
    AmountBucket,
    DisputeCase,
    EscalationTeam,
    EvalQuery,
    IssueFamily,
    MerchantCategory,
    MerchantSize,
    Outcome,
    PaymentRail,
    Region,
    Scheme,
)

REGION_TO_CURRENCY: dict[Region, str] = {
    Region.UK: "GBP",
    Region.EU: "EUR",
    Region.US: "USD",
    Region.SOUTH_AFRICA: "ZAR",
    Region.UAE: "AED",
}

MERCHANT_NAMES: dict[MerchantCategory, list[str]] = {
    MerchantCategory.DIGITAL_GOODS: [
        "Nimbus Media",
        "PixelForge",
        "Atlas Downloads",
        "Signal Arcade",
        "Velvet Streams",
    ],
    MerchantCategory.TRAVEL: [
        "BlueJet Travel",
        "Harbor Miles",
        "AeroNest",
        "Summit Routes",
        "Urban Voyage",
    ],
    MerchantCategory.SAAS: [
        "LedgerLoop",
        "Northstar Cloud",
        "CircuitOps",
        "Metric Harbor",
        "StackPilot",
    ],
    MerchantCategory.SUBSCRIPTION_SERVICES: [
        "BrightBox Club",
        "HomeCrate Plus",
        "DailyTheory",
        "Pulse Pantry",
        "Crest Collectives",
    ],
    MerchantCategory.RETAIL: [
        "Oak & Quartz",
        "Helio Market",
        "Anchor Cart",
        "Southline Goods",
        "Ribbon Supply",
    ],
    MerchantCategory.LOGISTICS: [
        "Parcel Harbor",
        "SwiftLane",
        "Orbit Fulfilment",
        "Delta Dispatch",
        "Cargo Bloom",
    ],
    MerchantCategory.FOOD_DELIVERY: [
        "DashDish",
        "MetroBite",
        "QuickPlate",
        "ForkSprint",
        "Basil Route",
    ],
    MerchantCategory.GAMING: [
        "LevelMint",
        "Arcbyte",
        "MoonToken Games",
        "Respawn Labs",
        "Quest Harbor",
    ],
}

MERCHANT_SIZES: list[MerchantSize] = [
    MerchantSize.STARTUP,
    MerchantSize.MID_MARKET,
    MerchantSize.ENTERPRISE,
]

CARD_PAIRS: list[tuple[PaymentRail, Scheme]] = [
    (PaymentRail.CARD, Scheme.VISA),
    (PaymentRail.CARD, Scheme.MASTERCARD),
    (PaymentRail.CARD, Scheme.AMEX),
]

WALLET_PAIRS: list[tuple[PaymentRail, Scheme]] = [
    (PaymentRail.WALLET, Scheme.PAYPAL),
    (PaymentRail.WALLET, Scheme.APPLE_PAY),
]

BANK_PAIRS: list[tuple[PaymentRail, Scheme]] = [
    (PaymentRail.BANK_TRANSFER, Scheme.SEPA),
]


@dataclass(frozen=True)
class IssueFamilyBlueprint:
    issue_family: IssueFamily
    allowed_pairs: list[tuple[PaymentRail, Scheme]]
    regions: list[Region]
    merchant_categories: list[MerchantCategory]
    amount_buckets: list[AmountBucket]
    reason_codes: list[str]
    title_templates: list[str]
    claim_templates: list[str]
    merchant_response_templates: list[str]
    processor_note_templates: list[str]
    resolution_templates: list[str]
    evidence_pool: list[str]
    risk_flag_pool: list[str]
    escalation_teams: list[EscalationTeam]
    outcome_weights: list[tuple[Outcome, int]]
    query_templates: list[str]


def _build_blueprints() -> list[IssueFamilyBlueprint]:
    return [
        IssueFamilyBlueprint(
            issue_family=IssueFamily.DUPLICATE_CAPTURE_AFTER_RETRY,
            allowed_pairs=CARD_PAIRS + WALLET_PAIRS,
            regions=[Region.UK, Region.EU, Region.US, Region.SOUTH_AFRICA],
            merchant_categories=[MerchantCategory.SAAS, MerchantCategory.RETAIL, MerchantCategory.DIGITAL_GOODS],
            amount_buckets=[AmountBucket.BETWEEN_25_AND_100, AmountBucket.BETWEEN_100_AND_500, AmountBucket.BETWEEN_500_AND_2000],
            reason_codes=["12.6", "13.2", "13.6"],
            title_templates=[
                "Duplicate debit after retry loop at {merchant}",
                "Second capture created after client retry on {scheme}",
                "Duplicate settlement dispute following payment retry",
            ],
            claim_templates=[
                "Customer says they were charged twice after the checkout retried and the second confirmation arrived late.",
                "Cardholder reports a duplicate debit after retrying the payment when the first attempt looked stuck.",
                "User claims the merchant triggered a second charge after timeout and manual retry.",
            ],
            merchant_response_templates=[
                "Merchant says the first authorization looked incomplete and the application retried before webhook confirmation landed.",
                "Merchant states the client retried checkout during a delayed provider response and both captures were later submitted.",
                "Merchant notes duplicate capture risk came from retry behavior and missing idempotency enforcement in one path.",
            ],
            processor_note_templates=[
                "Gateway logs show overlapping authorization and capture events within a short retry window.",
                "Internal notes mention the dispute contains duplicate language but may actually be a retry-idempotency failure rather than fraud.",
                "Processor trace indicates a second capture request was accepted before the original status callback completed.",
            ],
            resolution_templates=[
                "Merchant prevailed after providing capture timeline, retry logs, and proof of customer refund on the duplicate leg.",
                "Representment required idempotency evidence and event chronology to distinguish duplicate capture from delayed presentment.",
                "Case routed to platform disputes because the duplicate event aligned with retry handling in the orchestration layer.",
            ],
            evidence_pool=[
                "gateway retry trace",
                "authorization-to-capture timeline",
                "idempotency key inspection",
                "webhook delivery log",
                "merchant order ledger snapshot",
            ],
            risk_flag_pool=[
                "duplicate_keyword_overlap",
                "retry_path_without_idempotency",
                "late_webhook_confirmation",
                "false_positive_fraud_language",
            ],
            escalation_teams=[EscalationTeam.PAYMENTS_PLATFORM_DISPUTES, EscalationTeam.DISPUTES_OPS],
            outcome_weights=[
                (Outcome.MERCHANT_PREVAILED, 4),
                (Outcome.REPRESENTMENT_REQUIRED, 3),
                (Outcome.GOODWILL_REFUND, 2),
                (Outcome.CARDHOLDER_PREVAILED, 1),
            ],
            query_templates=[
                "Customer says there was a duplicate charge after retry during a timeout on {scheme}. Need similar cases from {region}.",
                "Looking for disputes where retry logic caused two debits for {merchant_category} payments on {scheme}.",
                "Find payment dispute cases with duplicate capture language, timeout symptoms, and reason code {reason_code}.",
            ],
        ),
        IssueFamilyBlueprint(
            issue_family=IssueFamily.THREE_DS_AUTHENTICATION_TIMEOUT,
            allowed_pairs=CARD_PAIRS,
            regions=[Region.UK, Region.EU, Region.US, Region.UAE],
            merchant_categories=[MerchantCategory.TRAVEL, MerchantCategory.RETAIL, MerchantCategory.SAAS],
            amount_buckets=[AmountBucket.BETWEEN_25_AND_100, AmountBucket.BETWEEN_100_AND_500, AmountBucket.BETWEEN_500_AND_2000],
            reason_codes=["10.4", "13.1", "13.3"],
            title_templates=[
                "3DS timeout dispute at {merchant}",
                "Authentication challenge timed out before payment completion",
                "Chargeback tied to 3DS challenge expiry on {scheme}",
            ],
            claim_templates=[
                "Customer says the authentication step froze and they are unsure whether the payment should have gone through.",
                "Cardholder claims the 3DS challenge timed out and the merchant still completed the charge.",
                "User reports a timeout during verification followed by a settled transaction they believed had failed.",
            ],
            merchant_response_templates=[
                "Merchant says issuer challenge completed too late and the platform received an authorization after customer abandonment.",
                "Merchant notes the 3DS session timed out visually, but issuer logs suggest the challenge result returned after redirect loss.",
                "Merchant claims the authorization was valid and the dispute is caused by customer confusion around the challenge timeout.",
            ],
            processor_note_templates=[
                "Processor notes contain timeout vocabulary that often overlaps with fraud and duplicate-charge complaints.",
                "Issuer ACS response timing suggests the user dropped before completion while the authorization still cleared.",
                "Internal trace shows partial challenge completion and a delayed browser redirect.",
            ],
            resolution_templates=[
                "Case required issuer challenge logs, ACS timing, and redirect telemetry before routing to fraud or disputes ops.",
                "Merchant prevailed after proving challenge completion and valid authorization despite timeout perception.",
                "Escalated to merchant integrations because the challenge UX created customer-abandonment confusion.",
            ],
            evidence_pool=[
                "3DS challenge log",
                "ACS response timeline",
                "redirect telemetry",
                "authorization approval record",
                "session expiry trace",
            ],
            risk_flag_pool=[
                "timeout_language_overlap",
                "abandoned_checkout_signal",
                "issuer_challenge_delay",
                "customer_confusion_risk",
            ],
            escalation_teams=[EscalationTeam.MERCHANT_INTEGRATIONS, EscalationTeam.DISPUTES_OPS],
            outcome_weights=[
                (Outcome.MERCHANT_PREVAILED, 3),
                (Outcome.REPRESENTMENT_REQUIRED, 3),
                (Outcome.CARDHOLDER_PREVAILED, 2),
                (Outcome.WRITE_OFF, 1),
            ],
            query_templates=[
                "Need similar payment disputes where a 3DS challenge timed out but the charge still settled on {scheme}.",
                "Find dispute cases about authentication timeout and customer abandonment in {region}.",
                "Which prior cases match a cardholder saying the verification page froze before charge completion?",
            ],
        ),
        IssueFamilyBlueprint(
            issue_family=IssueFamily.FRAUD_CARD_NOT_PRESENT,
            allowed_pairs=CARD_PAIRS,
            regions=[Region.UK, Region.EU, Region.US, Region.SOUTH_AFRICA, Region.UAE],
            merchant_categories=[MerchantCategory.RETAIL, MerchantCategory.TRAVEL, MerchantCategory.FOOD_DELIVERY],
            amount_buckets=[AmountBucket.BETWEEN_25_AND_100, AmountBucket.BETWEEN_100_AND_500, AmountBucket.BETWEEN_500_AND_2000, AmountBucket.ABOVE_2000],
            reason_codes=["10.4", "10.5", "10.9"],
            title_templates=[
                "Card-not-present fraud dispute for {merchant}",
                "Unauthorized remote purchase claim on {scheme}",
                "Fraud dispute with no cardholder recognition of transaction",
            ],
            claim_templates=[
                "Cardholder states they did not authorize the remote purchase and do not recognize the merchant.",
                "Customer claims the transaction was fraudulent and says the card was never used by them at this merchant.",
                "User reports an unauthorized e-commerce payment and denies any relationship with the merchant.",
            ],
            merchant_response_templates=[
                "Merchant submitted order, AVS, device, and fulfillment data to support legitimacy of the transaction.",
                "Merchant says the order passed fraud checks, matched account history, and shipped to a previously used address.",
                "Merchant argues the payment resembles a legitimate prior customer pattern rather than true third-party fraud.",
            ],
            processor_note_templates=[
                "Fraud language overlaps with 3DS timeout and descriptor mismatch cases when customers simply do not recognize the charge.",
                "Internal notes show standard remote-purchase indicators and incomplete customer recognition signals.",
                "Processor trace records normal authorization flow but elevated fraud scoring at post-transaction review.",
            ],
            resolution_templates=[
                "Case routed to fraud ops for deep review of device, velocity, and historical account linkage.",
                "Cardholder prevailed after merchant evidence failed to overcome unauthorized-use claim.",
                "Representment requested additional device graph evidence and prior customer relationship proof.",
            ],
            evidence_pool=[
                "device fingerprint",
                "AVS/CVV result",
                "historical account linkage",
                "fulfillment proof",
                "velocity review snapshot",
            ],
            risk_flag_pool=[
                "high_fraud_score",
                "customer_non_recognition",
                "cross_border_signal",
                "weak_post_auth_documentation",
            ],
            escalation_teams=[EscalationTeam.FRAUD_OPS, EscalationTeam.DISPUTES_OPS],
            outcome_weights=[
                (Outcome.CARDHOLDER_PREVAILED, 4),
                (Outcome.REPRESENTMENT_REQUIRED, 3),
                (Outcome.MERCHANT_PREVAILED, 2),
                (Outcome.WRITE_OFF, 1),
            ],
            query_templates=[
                "Find similar fraud card-not-present disputes where the customer says they do not recognize the merchant.",
                "Need prior unauthorized e-commerce payment cases on {scheme} in {region}.",
                "Looking for fraud disputes with remote purchase evidence and no cardholder recognition.",
            ],
        ),
        IssueFamilyBlueprint(
            issue_family=IssueFamily.FRIENDLY_FRAUD_DIGITAL_GOODS,
            allowed_pairs=CARD_PAIRS + WALLET_PAIRS,
            regions=[Region.UK, Region.EU, Region.US, Region.SOUTH_AFRICA],
            merchant_categories=[MerchantCategory.DIGITAL_GOODS, MerchantCategory.GAMING],
            amount_buckets=[AmountBucket.UNDER_25, AmountBucket.BETWEEN_25_AND_100, AmountBucket.BETWEEN_100_AND_500],
            reason_codes=["13.1", "13.3", "13.5"],
            title_templates=[
                "Digital goods charge disputed after content access at {merchant}",
                "Friendly fraud dispute on downloadable goods",
                "Chargeback on in-app purchase after successful delivery",
            ],
            claim_templates=[
                "Customer says the digital purchase was not received or should have been canceled, despite later usage.",
                "Cardholder disputes in-app or downloadable purchase after access logs show consumption.",
                "User claims the digital goods charge is invalid even though the account later used the purchased content.",
            ],
            merchant_response_templates=[
                "Merchant says the content was provisioned instantly and account activity confirms post-purchase usage.",
                "Merchant notes the customer accessed the digital entitlement after the dispute-triggering transaction.",
                "Merchant argues this looks like post-delivery chargeback behavior rather than true non-receipt.",
            ],
            processor_note_templates=[
                "Case language overlaps with refund and subscription disputes because customers often use non-receipt wording.",
                "Internal notes highlight strong entitlement evidence but weak physical-delivery analogies.",
                "Account telemetry indicates successful content redemption after payment settlement.",
            ],
            resolution_templates=[
                "Merchant prevailed after presenting entitlement, login, and post-purchase usage evidence.",
                "Representment succeeded because access logs contradicted the customer's non-receipt narrative.",
                "Case transferred to chargeback QA due to repeated friendly-fraud patterns on digital entitlements.",
            ],
            evidence_pool=[
                "entitlement grant log",
                "post-purchase usage log",
                "account login trail",
                "content redemption proof",
                "in-app purchase receipt",
            ],
            risk_flag_pool=[
                "friendly_fraud_pattern",
                "non_receipt_language_overlap",
                "repeat_disputer",
                "usage_after_chargeback",
            ],
            escalation_teams=[EscalationTeam.CHARGEBACK_QA, EscalationTeam.DISPUTES_OPS],
            outcome_weights=[
                (Outcome.MERCHANT_PREVAILED, 5),
                (Outcome.REPRESENTMENT_REQUIRED, 3),
                (Outcome.CARDHOLDER_PREVAILED, 1),
                (Outcome.WRITE_OFF, 1),
            ],
            query_templates=[
                "Need similar digital goods chargeback cases where content was accessed after purchase.",
                "Find disputes that look like friendly fraud for gaming or downloadable goods on {scheme}.",
                "Looking for prior cases with non-receipt language but clear post-purchase usage evidence.",
            ],
        ),
        IssueFamilyBlueprint(
            issue_family=IssueFamily.SUBSCRIPTION_CANCELLATION_CLAIM,
            allowed_pairs=CARD_PAIRS + WALLET_PAIRS,
            regions=[Region.UK, Region.EU, Region.US, Region.SOUTH_AFRICA],
            merchant_categories=[MerchantCategory.SUBSCRIPTION_SERVICES, MerchantCategory.SAAS],
            amount_buckets=[AmountBucket.UNDER_25, AmountBucket.BETWEEN_25_AND_100, AmountBucket.BETWEEN_100_AND_500],
            reason_codes=["13.2", "13.7", "13.5"],
            title_templates=[
                "Subscription cancellation dispute at {merchant}",
                "Recurring charge after claimed cancellation",
                "Cardholder says subscription should have stopped before billing",
            ],
            claim_templates=[
                "Customer says the subscription was canceled before renewal and the new charge should not have happened.",
                "Cardholder claims they ended the plan but still saw another recurring bill.",
                "User disputes a renewal charge and says cancellation had already been requested.",
            ],
            merchant_response_templates=[
                "Merchant says cancellation request came after the renewal cutoff window defined in the service terms.",
                "Merchant notes the customer remained active at renewal and cancellation timestamp was after invoice creation.",
                "Merchant argues the recurring charge followed contract timing and no valid pre-renewal cancellation existed.",
            ],
            processor_note_templates=[
                "Cancellation language often overlaps with refund-not-processed claims where customer expects an immediate reversal.",
                "Internal notes show contract timing and invoice generation occurred before the recorded cancellation event.",
                "Case requires careful separation between cancellation timing and refund processing expectations.",
            ],
            resolution_templates=[
                "Merchant prevailed after producing cancellation timestamp, renewal cutoff policy, and customer notifications.",
                "Representment required account timeline and cancellation audit trail before final routing.",
                "Goodwill refund issued despite merchant process being technically valid due to customer confusion risk.",
            ],
            evidence_pool=[
                "cancellation audit trail",
                "renewal notice email log",
                "billing cycle timeline",
                "terms acceptance record",
                "account activity at renewal",
            ],
            risk_flag_pool=[
                "renewal_confusion",
                "late_cancellation_request",
                "refund_language_overlap",
                "weak_notification_history",
            ],
            escalation_teams=[EscalationTeam.DISPUTES_OPS, EscalationTeam.CHARGEBACK_QA],
            outcome_weights=[
                (Outcome.MERCHANT_PREVAILED, 3),
                (Outcome.GOODWILL_REFUND, 3),
                (Outcome.REPRESENTMENT_REQUIRED, 2),
                (Outcome.CARDHOLDER_PREVAILED, 2),
            ],
            query_templates=[
                "Find subscription disputes where the customer says they canceled before renewal.",
                "Need similar recurring billing chargebacks with cancellation timing evidence.",
                "Looking for cases about renewal charges after claimed cancellation on {scheme}.",
            ],
        ),
        IssueFamilyBlueprint(
            issue_family=IssueFamily.REFUND_NOT_PROCESSED_CLAIM,
            allowed_pairs=CARD_PAIRS + WALLET_PAIRS + BANK_PAIRS,
            regions=[Region.UK, Region.EU, Region.US, Region.SOUTH_AFRICA, Region.UAE],
            merchant_categories=[MerchantCategory.RETAIL, MerchantCategory.TRAVEL, MerchantCategory.FOOD_DELIVERY],
            amount_buckets=[AmountBucket.UNDER_25, AmountBucket.BETWEEN_25_AND_100, AmountBucket.BETWEEN_100_AND_500, AmountBucket.BETWEEN_500_AND_2000],
            reason_codes=["13.6", "13.7", "12.6"],
            title_templates=[
                "Refund not received dispute at {merchant}",
                "Cardholder says merchant promised refund that never landed",
                "Chargeback after delayed refund processing",
            ],
            claim_templates=[
                "Customer says the merchant agreed to refund the transaction but the money never returned.",
                "Cardholder disputes because a promised reversal did not appear on the statement in time.",
                "User claims refund was approved but still missing after the expected settlement window.",
            ],
            merchant_response_templates=[
                "Merchant says the refund was initiated but cardholder is still within issuer settlement timeframes.",
                "Merchant notes the refund reference exists and any delay is downstream of scheme or issuer posting.",
                "Merchant argues the reversal was processed correctly and the dispute confuses refund initiation with posting date.",
            ],
            processor_note_templates=[
                "Refund language often overlaps with subscription and partial-shipment disputes when customers expect immediate reversal.",
                "Internal notes show refund reference present but posting window may vary by issuer and scheme.",
                "Case may look like non-receipt despite successful merchant-side refund initiation.",
            ],
            resolution_templates=[
                "Merchant prevailed after showing refund reference and settlement timing consistent with scheme rules.",
                "Goodwill refund or outreach issued when posting delay caused predictable customer confusion.",
                "Case escalated to disputes ops to separate merchant refund initiation from issuer posting latency.",
            ],
            evidence_pool=[
                "refund reference ID",
                "refund initiation timestamp",
                "statement descriptor mapping",
                "issuer posting window note",
                "merchant CRM refund confirmation",
            ],
            risk_flag_pool=[
                "refund_posting_delay",
                "customer_expectation_gap",
                "issuer_visibility_lag",
                "language_overlap_with_cancellation",
            ],
            escalation_teams=[EscalationTeam.DISPUTES_OPS, EscalationTeam.MERCHANT_INTEGRATIONS],
            outcome_weights=[
                (Outcome.MERCHANT_PREVAILED, 3),
                (Outcome.GOODWILL_REFUND, 3),
                (Outcome.REPRESENTMENT_REQUIRED, 2),
                (Outcome.CARDHOLDER_PREVAILED, 2),
            ],
            query_templates=[
                "Need similar disputes where the refund was initiated but the customer says nothing came back.",
                "Find payment disputes involving delayed refund posting for {scheme} in {region}.",
                "Looking for cases where refund language masks a settlement-window delay instead of a missing refund.",
            ],
        ),
        IssueFamilyBlueprint(
            issue_family=IssueFamily.CHARGEBACK_MISSING_DESCRIPTOR,
            allowed_pairs=CARD_PAIRS + WALLET_PAIRS,
            regions=[Region.UK, Region.EU, Region.US, Region.SOUTH_AFRICA],
            merchant_categories=[MerchantCategory.RETAIL, MerchantCategory.FOOD_DELIVERY, MerchantCategory.TRAVEL],
            amount_buckets=[AmountBucket.UNDER_25, AmountBucket.BETWEEN_25_AND_100, AmountBucket.BETWEEN_100_AND_500],
            reason_codes=["13.1", "12.5", "13.6"],
            title_templates=[
                "Customer does not recognize descriptor for {merchant}",
                "Unrecognized merchant descriptor dispute",
                "Chargeback tied to soft descriptor mismatch",
            ],
            claim_templates=[
                "Customer says they do not recognize the statement descriptor and believe the charge is unauthorized.",
                "Cardholder disputes because the descriptor on the statement does not match the storefront brand they know.",
                "User claims the charge is unfamiliar due to merchant naming mismatch on the statement.",
            ],
            merchant_response_templates=[
                "Merchant says the legal entity descriptor differs from the consumer-facing brand but represents the same purchase.",
                "Merchant notes descriptor truncation or aggregator naming likely caused customer non-recognition.",
                "Merchant argues the transaction is valid and the confusion comes from statement naming rather than fraud.",
            ],
            processor_note_templates=[
                "Descriptor confusion overlaps heavily with fraud and merchant-mismatch cases in retrieval.",
                "Internal notes show branding mismatch between checkout page and statement descriptor.",
                "Case requires separation between true unauthorized use and descriptor recognition failure.",
            ],
            resolution_templates=[
                "Merchant prevailed after supplying brand-to-descriptor mapping and proof of customer purchase context.",
                "Case routed to merchant integrations because descriptor configuration increased avoidable disputes.",
                "Goodwill refund applied for low-value transactions where descriptor confusion was likely.",
            ],
            evidence_pool=[
                "descriptor-to-brand mapping",
                "checkout receipt",
                "brand ownership proof",
                "support interaction transcript",
                "statement sample",
            ],
            risk_flag_pool=[
                "descriptor_confusion",
                "fraud_language_overlap",
                "soft_descriptor_gap",
                "aggregator_naming_issue",
            ],
            escalation_teams=[EscalationTeam.MERCHANT_INTEGRATIONS, EscalationTeam.DISPUTES_OPS],
            outcome_weights=[
                (Outcome.MERCHANT_PREVAILED, 4),
                (Outcome.GOODWILL_REFUND, 3),
                (Outcome.CARDHOLDER_PREVAILED, 2),
                (Outcome.WRITE_OFF, 1),
            ],
            query_templates=[
                "Find disputes where the customer did not recognize the descriptor but the transaction may still be valid.",
                "Need prior chargebacks caused by descriptor mismatch or storefront-brand confusion.",
                "Looking for cases with unfamiliar statement text on {scheme} rather than true fraud.",
            ],
        ),
        IssueFamilyBlueprint(
            issue_family=IssueFamily.MERCHANT_DESCRIPTOR_MISMATCH,
            allowed_pairs=CARD_PAIRS + WALLET_PAIRS,
            regions=[Region.UK, Region.EU, Region.US, Region.SOUTH_AFRICA],
            merchant_categories=[MerchantCategory.DIGITAL_GOODS, MerchantCategory.SUBSCRIPTION_SERVICES, MerchantCategory.RETAIL],
            amount_buckets=[AmountBucket.UNDER_25, AmountBucket.BETWEEN_25_AND_100, AmountBucket.BETWEEN_100_AND_500],
            reason_codes=["12.5", "13.1", "13.3"],
            title_templates=[
                "Brand mismatch between checkout and statement at {merchant}",
                "Merchant alias mismatch dispute",
                "Consumer-facing brand differs from acquirer descriptor",
            ],
            claim_templates=[
                "Customer says the statement name is different from the brand they thought they were paying.",
                "Cardholder disputes because the merchant alias on the statement does not resemble the checkout brand.",
                "User claims the charge is suspicious due to a mismatch between on-site branding and statement text.",
            ],
            merchant_response_templates=[
                "Merchant says the checkout brand operates under a parent descriptor and the mismatch is expected but poorly surfaced.",
                "Merchant notes affiliate or reseller naming caused the charge to post under a different legal descriptor.",
                "Merchant argues the payment is valid and the issue is brand aliasing rather than non-delivery or fraud.",
            ],
            processor_note_templates=[
                "This family is intentionally close to missing descriptor and fraud cases to stress retrieval quality.",
                "Internal notes show merchant aliasing, reseller relationships, or parent-entity posting behavior.",
                "Descriptor mismatch language can look nearly identical to unrecognized-fraud complaints.",
            ],
            resolution_templates=[
                "Merchant prevailed after providing alias mapping and transaction context tying the descriptor to the storefront.",
                "Case escalated to merchant integrations to improve statement branding and reduce avoidable disputes.",
                "Goodwill refund used where consumer-facing naming was materially misleading.",
            ],
            evidence_pool=[
                "alias mapping document",
                "merchant legal entity proof",
                "checkout branding screenshot",
                "parent-brand relationship note",
                "descriptor configuration export",
            ],
            risk_flag_pool=[
                "brand_alias_overlap",
                "descriptor_confusion",
                "fraud_language_overlap",
                "reseller_model_complexity",
            ],
            escalation_teams=[EscalationTeam.MERCHANT_INTEGRATIONS, EscalationTeam.CHARGEBACK_QA],
            outcome_weights=[
                (Outcome.MERCHANT_PREVAILED, 3),
                (Outcome.GOODWILL_REFUND, 3),
                (Outcome.REPRESENTMENT_REQUIRED, 2),
                (Outcome.CARDHOLDER_PREVAILED, 2),
            ],
            query_templates=[
                "Need similar cases where the checkout brand and statement descriptor were different.",
                "Find payment disputes tied to parent-company or alias descriptor mismatch.",
                "Looking for retrieval cases where merchant aliasing looks like fraud but is operationally valid.",
            ],
        ),
        IssueFamilyBlueprint(
            issue_family=IssueFamily.DELAYED_PRESENTMENT_CONFUSION,
            allowed_pairs=CARD_PAIRS + WALLET_PAIRS,
            regions=[Region.UK, Region.EU, Region.US, Region.SOUTH_AFRICA, Region.UAE],
            merchant_categories=[MerchantCategory.TRAVEL, MerchantCategory.LOGISTICS, MerchantCategory.FOOD_DELIVERY],
            amount_buckets=[AmountBucket.BETWEEN_25_AND_100, AmountBucket.BETWEEN_100_AND_500, AmountBucket.BETWEEN_500_AND_2000],
            reason_codes=["12.6", "13.6", "13.3"],
            title_templates=[
                "Delayed presentment mistaken for duplicate charge at {merchant}",
                "Customer disputes late-captured transaction",
                "Late posting created duplicate-charge confusion",
            ],
            claim_templates=[
                "Customer says an old authorization appeared to become a second charge days later.",
                "Cardholder disputes a late-posted transaction they believed had already dropped off.",
                "User claims a delayed capture created the appearance of a duplicate or unexpected new charge.",
            ],
            merchant_response_templates=[
                "Merchant says the authorization and presentment were separated by normal operational delay for the service type.",
                "Merchant notes the transaction posted later due to fulfillment or travel settlement workflow.",
                "Merchant argues the cardholder confused delayed presentment with a second charge.",
            ],
            processor_note_templates=[
                "This family intentionally overlaps with duplicate-capture language but differs in event chronology.",
                "Internal notes show a single economic event with delayed presentment rather than duplicate debit.",
                "Timeline analysis is required to separate retry failures from presentment lag.",
            ],
            resolution_templates=[
                "Merchant prevailed after presentment timing evidence proved the charge was delayed rather than duplicated.",
                "Representment required event chronology and fulfillment timing to separate duplicate and delayed presentment cases.",
                "Case routed to disputes ops because delay windows varied by service and confused customer expectations.",
            ],
            evidence_pool=[
                "authorization-presentment timeline",
                "fulfillment event log",
                "travel or delivery completion proof",
                "merchant settlement batch record",
                "customer receipt timestamp",
            ],
            risk_flag_pool=[
                "duplicate_language_overlap",
                "late_presentment_confusion",
                "timeline_analysis_required",
                "service_fulfillment_dependency",
            ],
            escalation_teams=[EscalationTeam.DISPUTES_OPS, EscalationTeam.PAYMENTS_PLATFORM_DISPUTES],
            outcome_weights=[
                (Outcome.MERCHANT_PREVAILED, 4),
                (Outcome.REPRESENTMENT_REQUIRED, 3),
                (Outcome.CARDHOLDER_PREVAILED, 1),
                (Outcome.GOODWILL_REFUND, 2),
            ],
            query_templates=[
                "Find disputes where a delayed presentment looked like a duplicate charge.",
                "Need similar cases involving late capture or posting on {scheme}.",
                "Looking for one-charge-two-events confusion: old auth plus later settlement.",
            ],
        ),
        IssueFamilyBlueprint(
            issue_family=IssueFamily.PARTIAL_SHIPMENT_CLAIM,
            allowed_pairs=CARD_PAIRS + WALLET_PAIRS + BANK_PAIRS,
            regions=[Region.UK, Region.EU, Region.US, Region.SOUTH_AFRICA],
            merchant_categories=[MerchantCategory.RETAIL, MerchantCategory.LOGISTICS],
            amount_buckets=[AmountBucket.BETWEEN_25_AND_100, AmountBucket.BETWEEN_100_AND_500, AmountBucket.BETWEEN_500_AND_2000],
            reason_codes=["13.1", "13.3", "13.7"],
            title_templates=[
                "Partial shipment dispute at {merchant}",
                "Customer claims only part of the order was delivered",
                "Chargeback tied to incomplete fulfillment",
            ],
            claim_templates=[
                "Customer says only part of the order arrived and disputes the full transaction amount.",
                "Cardholder claims the merchant delivered a partial shipment but charged the complete basket value.",
                "User reports incomplete fulfillment and seeks reversal for the undelivered portion.",
            ],
            merchant_response_templates=[
                "Merchant says the order was split across shipments and the remaining items were still in transit.",
                "Merchant notes the charge covered the full order and partial delivery was expected operationally.",
                "Merchant argues the customer disputed before the final shipment or compensation workflow completed.",
            ],
            processor_note_templates=[
                "Language overlaps with refund and non-receipt cases, but fulfillment chronology is the real differentiator.",
                "Internal notes show split-shipment logic and tracking events that can be confused with non-delivery.",
                "Case may require separating partial shipment from refund-not-processed expectations.",
            ],
            resolution_templates=[
                "Representment required shipment tracking, split-order evidence, and compensation timeline.",
                "Goodwill refund used when operational split shipment was valid but communication was weak.",
                "Merchant prevailed after proving the order was fulfilled in multiple legs with transparent tracking.",
            ],
            evidence_pool=[
                "split-shipment tracking log",
                "fulfillment record",
                "delivery confirmation",
                "customer support transcript",
                "replacement or refund workflow note",
            ],
            risk_flag_pool=[
                "split_fulfillment_confusion",
                "refund_language_overlap",
                "non_receipt_overlap",
                "late_customer_support_response",
            ],
            escalation_teams=[EscalationTeam.DISPUTES_OPS, EscalationTeam.MERCHANT_INTEGRATIONS],
            outcome_weights=[
                (Outcome.REPRESENTMENT_REQUIRED, 3),
                (Outcome.GOODWILL_REFUND, 3),
                (Outcome.MERCHANT_PREVAILED, 2),
                (Outcome.CARDHOLDER_PREVAILED, 2),
            ],
            query_templates=[
                "Need similar disputes where only part of the order shipped but the full amount was charged.",
                "Find payment disputes about split shipments, incomplete delivery, or undelivered item claims.",
                "Looking for cases where partial fulfillment was confused with full non-receipt.",
            ],
        ),
    ]


class SyntheticDisputeDataGenerator:
    def __init__(
        self,
        seed: int = 42,
        reference_time: datetime | None = None,
    ) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.blueprints = _build_blueprints()
        self.blueprint_by_family = {bp.issue_family: bp for bp in self.blueprints}
        self.reference_time = reference_time or datetime(2026, 1, 1, tzinfo=UTC)

    def generate_cases(self, case_count: int) -> list[DisputeCase]:
        if case_count < len(self.blueprints):
            raise ValueError(
                f"case_count must be at least {len(self.blueprints)} to cover every issue family"
            )

        sequence = self._build_blueprint_sequence(case_count)
        cases: list[DisputeCase] = []

        for index, blueprint in enumerate(sequence, start=1):
            cases.append(self._generate_case(index=index, blueprint=blueprint))

        return cases

    def generate_eval_queries(
        self,
        cases: list[DisputeCase],
        eval_query_count: int,
    ) -> list[EvalQuery]:
        if not cases:
            raise ValueError("cases must not be empty")

        grouped_by_family: dict[IssueFamily, list[DisputeCase]] = defaultdict(list)
        for case in cases:
            grouped_by_family[case.issue_family].append(case)

        for family_cases in grouped_by_family.values():
            family_cases.sort(key=lambda item: item.created_at, reverse=True)

        eval_queries: list[EvalQuery] = []
        case_pool = list(cases)
        self.rng.shuffle(case_pool)

        for index in range(1, eval_query_count + 1):
            anchor = case_pool[(index - 1) % len(case_pool)]
            blueprint = self.blueprint_by_family[anchor.issue_family]

            candidate_gold = [
                case
                for case in grouped_by_family[anchor.issue_family]
                if case.payment_rail == anchor.payment_rail
                and case.scheme == anchor.scheme
            ]
            if len(candidate_gold) < 3:
                candidate_gold = grouped_by_family[anchor.issue_family]

            gold_ids = [case.dispute_id for case in candidate_gold[:5]]

            query_text = self._render_query_text(anchor=anchor, blueprint=blueprint)

            eval_queries.append(
                EvalQuery(
                    query_id=f"Q-{index:04d}",
                    query_text=query_text,
                    filters={
                        "payment_rail": anchor.payment_rail.value,
                        "scheme": anchor.scheme.value,
                        "region": anchor.region.value,
                    },
                    gold_issue_family=anchor.issue_family,
                    gold_relevant_dispute_ids=gold_ids,
                    gold_escalation_team=anchor.escalation_team,
                )
            )

        return eval_queries

    def _build_blueprint_sequence(self, case_count: int) -> list[IssueFamilyBlueprint]:
        repeated = case_count // len(self.blueprints)
        remainder = case_count % len(self.blueprints)

        sequence = self.blueprints * repeated + self.rng.sample(self.blueprints, remainder)
        self.rng.shuffle(sequence)
        return sequence

    def _generate_case(self, index: int, blueprint: IssueFamilyBlueprint) -> DisputeCase:
        payment_rail, scheme = self.rng.choice(blueprint.allowed_pairs)
        region = self.rng.choice(blueprint.regions)
        merchant_category = self.rng.choice(blueprint.merchant_categories)
        amount_bucket = self.rng.choice(blueprint.amount_buckets)
        merchant_name = self._merchant_name(merchant_category)
        merchant_size = self.rng.choice(MERCHANT_SIZES)
        reason_code = self.rng.choice(blueprint.reason_codes)
        currency = REGION_TO_CURRENCY[region]

        case_title = self.rng.choice(blueprint.title_templates).format(
            merchant=merchant_name,
            scheme=scheme.value.upper(),
        )
        customer_claim = self.rng.choice(blueprint.claim_templates)
        merchant_response_summary = self.rng.choice(blueprint.merchant_response_templates)
        processor_notes = self.rng.choice(blueprint.processor_note_templates)
        resolution_summary = self.rng.choice(blueprint.resolution_templates)
        evidence_submitted = self._sample_nonempty(blueprint.evidence_pool, minimum=2, maximum=4)
        risk_flags = self._sample_nonempty(blueprint.risk_flag_pool, minimum=1, maximum=3)
        escalation_team = self.rng.choice(blueprint.escalation_teams)
        outcome = self._weighted_choice(blueprint.outcome_weights)
        created_at = self._random_datetime()

        case_summary = (
            f"Customer claim: {customer_claim} "
            f"Merchant response: {merchant_response_summary} "
            f"Operational context: {merchant_name} is a {merchant_size.value} "
            f"{merchant_category.value} merchant in {region.value}. "
            f"Reason code {reason_code} was attached to a {scheme.value} "
            f"{payment_rail.value} payment in bucket {amount_bucket.value}. "
            f"Processor notes: {processor_notes}"
        )

        return DisputeCase(
            dispute_id=f"D-{index:05d}",
            case_title=case_title,
            case_summary=case_summary,
            customer_claim=customer_claim,
            merchant_response_summary=merchant_response_summary,
            processor_notes=processor_notes,
            evidence_submitted=evidence_submitted,
            resolution_summary=resolution_summary,
            issue_family=blueprint.issue_family,
            reason_code=reason_code,
            scheme=scheme,
            payment_rail=payment_rail,
            region=region,
            merchant_name=merchant_name,
            merchant_category=merchant_category,
            merchant_size=merchant_size,
            amount_bucket=amount_bucket,
            currency=currency,
            risk_flags=risk_flags,
            outcome=outcome,
            escalation_team=escalation_team,
            created_at=created_at,
        )

    def _render_query_text(self, anchor: DisputeCase, blueprint: IssueFamilyBlueprint) -> str:
        template = self.rng.choice(blueprint.query_templates)
        return template.format(
            merchant=anchor.merchant_name,
            merchant_category=anchor.merchant_category.value.replace("_", " "),
            scheme=anchor.scheme.value.upper(),
            region=anchor.region.value.replace("_", " "),
            reason_code=anchor.reason_code,
        )

    def _merchant_name(self, category: MerchantCategory) -> str:
        return self.rng.choice(MERCHANT_NAMES[category])

    def _random_datetime(self) -> datetime:
        days_back = self.rng.randint(5, 730)
        minutes_back = self.rng.randint(0, 24 * 60 - 1)
        return self.reference_time - timedelta(days=days_back, minutes=minutes_back)

    def _sample_nonempty(
        self,
        items: list[str],
        minimum: int,
        maximum: int,
    ) -> list[str]:
        upper = min(len(items), maximum)
        lower = min(minimum, upper)
        sample_size = self.rng.randint(lower, upper)
        return self.rng.sample(items, sample_size)

    def _weighted_choice(self, weighted_items: list[tuple[Outcome, int]]) -> Outcome:
        population = [item for item, _ in weighted_items]
        weights = [weight for _, weight in weighted_items]
        return self.rng.choices(population=population, weights=weights, k=1)[0]


def write_cases_jsonl(cases: Iterable[DisputeCase], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n")


def write_cases_csv(cases: Iterable[DisputeCase], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases_list = list(cases)
    if not cases_list:
        raise ValueError("cases must not be empty")

    fieldnames = list(cases_list[0].model_dump(mode="json").keys())

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases_list:
            row = case.model_dump(mode="json")
            for key in ("evidence_submitted", "risk_flags"):
                row[key] = "|".join(row[key])
            writer.writerow(row)


def write_eval_queries_jsonl(eval_queries: Iterable[EvalQuery], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for query in eval_queries:
            handle.write(json.dumps(query.model_dump(mode="json"), ensure_ascii=False) + "\n")