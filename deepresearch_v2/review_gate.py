"""M2.4 explicit human-review packet and fail-closed comparison-readiness gate.

No API calls, automated semantic labels, current-pricing inference, or published
report generation. The gate reflects only user-supplied labels and binds them
to the exact report.json bytes. It cannot independently authenticate a reviewer.
"""
from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .core import ResearchReport
from .quality_flags import flag_claims_for_review

PassageSupport = Literal["supported", "contradicted", "insufficient", "uncertain"]
FieldFit = Literal["appropriate", "inappropriate", "uncertain"]
TemporalRelevance = Literal["current", "historical", "unknown", "not_time_sensitive"]


class ReviewDecision(BaseModel):
    claim_id: str = Field(min_length=1)
    passage_support: PassageSupport | None = None
    field_fit: FieldFit | None = None
    temporal_relevance: TemporalRelevance | None = None
    ready_for_comparison: bool | None = None
    note: str = Field(default="", max_length=2000)
    addressed_trigger_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def nonduplicate_flags(self) -> "ReviewDecision":
        if len(set(self.addressed_trigger_codes)) != len(self.addressed_trigger_codes):
            raise ValueError("Duplicate addressed trigger code")
        return self


class ReviewManifest(BaseModel):
    reviewer: str = Field(default="", max_length=200)
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions: list[ReviewDecision]

    @model_validator(mode="after")
    def distinct_ids(self) -> "ReviewManifest":
        ids = [d.claim_id for d in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate claim IDs in review decisions")
        return self


def report_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_review_packet(report: ResearchReport, report_bytes: bytes) -> dict:
    """Expose every extracted assertion with its full evidence and review cues."""
    source_map = {s.id: s for s in report.sources}
    evidence_map = {e.id: e for e in report.evidence}
    flags = flag_claims_for_review(report)
    by_claim: dict[str, list[dict]] = {}
    for flag in flags.flags:
        by_claim.setdefault(flag.claim_id, []).append(flag.model_dump(mode="json"))
    records = []
    for c in report.claims:
        refs = []
        for eid in c.evidence_ids:
            e = evidence_map[eid]
            source = source_map[e.source_id]
            refs.append({
                "evidence_id": eid,
                "source_id": e.source_id,
                "quote": e.quote,
                "source_title": source.title,
                "source_url": str(source.url),
                "retrieved_at": source.retrieved_at.isoformat(),
                "publication_date": None,  # Unsupported by M2.3 Source schema.
            })
        records.append({
            "claim_id": c.id, "company": c.company,
            "field": c.field, "value": c.value,
            "evidence": refs,
            "review_triggers": by_claim.get(c.id, []),
        })
    return {
        "schema_version": "m2.4-review-packet-v1",
        "report_sha256": report_sha256(report_bytes),
        "claims_total": len(records),
        "claims": records,
        "warning": (
            "For independent human review: source-reported statements only. "
            "Retrieval timestamp is not publication date; review triggers are not verdicts."
        ),
    }


def build_review_template(report: ResearchReport, report_bytes: bytes) -> dict:
    """All labels start NULL; no fabricated human sign-off."""
    return ReviewManifest(
        report_sha256=report_sha256(report_bytes),
        decisions=[ReviewDecision(claim_id=c.id) for c in report.claims],
    ).model_dump(mode="json")


def evaluate_review_gate(
    report: ResearchReport, report_bytes: bytes, manifest: ReviewManifest
) -> dict:
    """Complete human input required; ambiguous or inconsistent decisions block."""
    if manifest.report_sha256 != report_sha256(report_bytes):
        raise ValueError("Review manifest does not match exact report.json SHA-256")
    known = {c.id for c in report.claims}
    supplied = {d.claim_id for d in manifest.decisions}
    if supplied - known:
        raise ValueError(f"Unknown reviewed claim IDs: {sorted(supplied-known)}")
    triggers = flag_claims_for_review(report)
    by_claim: dict[str, set[str]] = {}
    for flag in triggers.flags:
        by_claim.setdefault(flag.claim_id, set()).add(flag.code)
    blockers: list[dict[str, str]] = []
    ready_ids = []
    for claim in report.claims:
        decision = next((d for d in manifest.decisions if d.claim_id == claim.id), None)
        if decision is None:
            blockers.append({"claim_id": claim.id, "reason": "missing_review"})
            continue
        if any(x is None for x in (
            decision.passage_support, decision.field_fit,
            decision.temporal_relevance, decision.ready_for_comparison
        )):
            blockers.append({"claim_id": claim.id, "reason": "incomplete_review"})
            continue
        if not decision.note.strip():
            blockers.append({"claim_id": claim.id, "reason": "missing_rationale"})
            continue
        expected = by_claim.get(claim.id, set())
        provided = set(decision.addressed_trigger_codes)
        if provided - expected:
            blockers.append({"claim_id": claim.id, "reason": "unexpected_trigger_acknowledgement"})
            continue
        if decision.ready_for_comparison:
            if (decision.passage_support != "supported"
                    or decision.field_fit != "appropriate"
                    or decision.temporal_relevance not in ("current", "not_time_sensitive")):
                blockers.append({"claim_id": claim.id, "reason": "invalid_ready_decision"})
                continue
            if provided != expected:
                blockers.append({"claim_id": claim.id, "reason": "unaddressed_review_triggers"})
                continue
            ready_ids.append(claim.id)
    if not manifest.reviewer.strip():
        blockers.append({"claim_id": "*", "reason": "missing_reviewer"})
    completed = len(blockers) == 0
    return {
        "schema_version": "m2.4-review-gate-v1",
        "report_sha256": manifest.report_sha256,
        "gate_status": "review_complete" if completed else "blocked",
        "reviewer_declared": manifest.reviewer if manifest.reviewer.strip() else None,
        "claims_total": len(report.claims),
        "decisions_supplied": len(manifest.decisions),
        "comparison_ready_claim_ids": ready_ids if completed else [],
        "comparison_ready_count": len(ready_ids) if completed else 0,
        "blockers": blockers,
        "review_basis": "user_supplied_decisions_not_independently_authenticated",
        "publication_generated": False,
        "warning": (
            "This gate validates review completeness/consistency, not factual accuracy. "
            "A self-reported reviewer identity does not prove independent human review."
        ),
    }
