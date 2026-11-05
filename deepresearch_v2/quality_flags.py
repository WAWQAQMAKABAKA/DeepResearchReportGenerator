"""M2.3 deterministic *review triggers*, NOT semantic or temporal proof.

Rules flag suspicious field use, standalone price amounts and explicit historical
source titles. They never reject or rewrite a lexically accepted claim, assert
truth, infer a publication date from retrieved_at, or mark a claim verified.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from .core import ResearchReport

RULES_VERSION = "m2.3-review-triggers-v1"
FlagCode = Literal[
    "pricing_amount_without_tier",
    "eligibility_as_target_customer",
    "use_case_as_target_customer",
    "historical_source_title",
]
FlagBasis = Literal["claim_value", "source_title"]

_PRICE_WITHOUT_TIER = re.compile(r"^\s*(?:[$€£]|(?:USD|EUR|GBP)\s*\d)", re.IGNORECASE)
_ELIGIBILITY = re.compile(
    r"^\s*(?:all\s+)?paid\b.{0,60}\bsubscribers?\b", re.IGNORECASE
)
_GENERIC_USE_CASE = re.compile(
    r"^\s*for\s+(?:everyday|daily|general)\s+(?:coding|development|work)\b",
    re.IGNORECASE,
)
_HISTORICAL_TITLE = re.compile(
    r"\b(?:historical|archived|archive|superseded)\b|"
    r"\bnot\s+(?:a\s+)?current\b",
    re.IGNORECASE,
)

_DESCRIPTIONS = {
    "pricing_amount_without_tier": (
        "Price starts with an amount, without an identified plan/tier in the "
        "claim value. Verify plan, currency, unit, billing basis and as-of date."
    ),
    "eligibility_as_target_customer": (
        "Paid subscriber eligibility describes access, not necessarily a "
        "target customer segment. Review field classification."
    ),
    "use_case_as_target_customer": (
        "The value describes a generic use case, not a customer segment. "
        "Review field classification."
    ),
    "historical_source_title": (
        "The captured source title explicitly signals historical context. "
        "Do not infer current validity; verify publication and effective dates."
    ),
}


class ReviewTrigger(BaseModel):
    claim_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    code: FlagCode
    basis: FlagBasis
    note: str


class ReviewTriggerReport(BaseModel):
    rules_version: str
    assessment: Literal["deterministic_review_triggers_not_verdicts"]
    total_claims: int
    flagged_claims: int
    flag_count: int
    by_code: dict[str, int]
    flags: list[ReviewTrigger]
    warning: str


def flag_claims_for_review(report: ResearchReport) -> ReviewTriggerReport:
    """Review hints for accepted claims; absence of flags NEVER means support."""
    sources = {s.id: s for s in report.sources}
    evidence = {e.id: e for e in report.evidence}
    flags: list[ReviewTrigger] = []
    seen: set[tuple[str, str, str]] = set()

    def add(claim_id: str, source_id: str, code: FlagCode, basis: FlagBasis) -> None:
        key = (claim_id, source_id, code)
        if key not in seen:
            seen.add(key)
            flags.append(ReviewTrigger(
                claim_id=claim_id, source_id=source_id, code=code,
                basis=basis, note=_DESCRIPTIONS[code]
            ))

    for claim in report.claims:
        source_ids = [evidence[eid].source_id for eid in claim.evidence_ids]
        if claim.field == "pricing" and _PRICE_WITHOUT_TIER.search(claim.value):
            add(claim.id, source_ids[0], "pricing_amount_without_tier", "claim_value")
        if claim.field == "target_customers":
            if _ELIGIBILITY.search(claim.value):
                add(claim.id, source_ids[0], "eligibility_as_target_customer", "claim_value")
            if _GENERIC_USE_CASE.search(claim.value):
                add(claim.id, source_ids[0], "use_case_as_target_customer", "claim_value")
        for source_id in source_ids:
            if _HISTORICAL_TITLE.search(sources[source_id].title):
                add(claim.id, source_id, "historical_source_title", "source_title")

    counts: dict[str, int] = {}
    for flag in flags:
        counts[flag.code] = counts.get(flag.code, 0) + 1
    return ReviewTriggerReport(
        rules_version=RULES_VERSION,
        assessment="deterministic_review_triggers_not_verdicts",
        total_claims=len(report.claims),
        flagged_claims=len({flag.claim_id for flag in flags}),
        flag_count=len(flags),
        by_code=dict(sorted(counts.items())),
        flags=flags,
        warning=(
            "Heuristic review triggers, not entailment checks or stale-price detection. "
            "Unflagged and lexically accepted claims remain UNREVIEWED."
        ),
    )
