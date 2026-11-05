"""M2.7: deterministic, inspectable preview of a captured company research report.

This renders source-reported statements, missing comparison fields, conflicts,
and user-declared review readiness WITHOUT upgrading lexical evidence to truth.
It is a non-publishable preview, not a semantic verification or report approval.
No model calls, no charts, no Streamlit dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .core import FIELDS, FieldName, ResearchReport
from .quality_flags import flag_claims_for_review
from .review_gate import ReviewManifest, evaluate_review_gate, report_sha256

FieldStatus = Literal[
    "not_established", "unresolved_conflict", "source_reported_unreviewed",
    "review_declared_ready", "review_declared_excluded", "mixed_review_decisions",
]
ReviewStatus = Literal["not_provided", "blocked", "review_complete"]
SCHEMA_VERSION = "m2.7-evidence-preview-v1"


class EvidenceLink(BaseModel):
    evidence_id: str
    source_id: str
    quote: str
    source_title: str
    source_url: str
    retrieved_at: str
    publication_date: None = None


class ClaimRow(BaseModel):
    id: str
    value: str
    citations: list[EvidenceLink]
    review_trigger_codes: list[str]
    source_reporting_status: Literal["lexically_linked_semantically_unverified"] = (
        "lexically_linked_semantically_unverified"
    )


class FieldRow(BaseModel):
    company: str
    field: FieldName
    status: FieldStatus
    values: list[ClaimRow]
    note: str


class EvidencePreview(BaseModel):
    schema_version: Literal["m2.7-evidence-preview-v1"] = SCHEMA_VERSION
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question: str
    companies: list[str]
    fields_per_company: int
    comparison_rows: list[FieldRow]
    status_counts: dict[str, int]
    source_reported_claims: int
    rejected_candidates: int
    calculated_observations: list[dict] = Field(default_factory=list)
    calculation_status: Literal["not_supported_by_current_report_schema"] = (
        "not_supported_by_current_report_schema"
    )
    review_status: ReviewStatus
    review_basis: str
    publication_ready: Literal[False] = False
    semantic_support_automatically_verified: Literal[False] = False
    warning: str = (
        "Lexical quote linkage is not factual entailment; review readiness "
        "is user-declared, not independently authenticated. No calculations "
        "are inferred from unstructured company claims. Do not publish."
    )


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _report_from_exact_bytes(report_bytes: bytes) -> ResearchReport:
    return ResearchReport.model_validate_json(report_bytes)


def build_preview(
    report: ResearchReport,
    report_bytes: bytes,
    manifest: ReviewManifest | None = None,
) -> EvidencePreview:
    """Recompose every preview cell from validated claims, never trust old table HTML."""
    report_from_bytes = _report_from_exact_bytes(report_bytes)
    if report.model_dump(mode="json") != report_from_bytes.model_dump(mode="json"):
        raise ValueError("Report object differs from exact report.json bytes")
    gate = (evaluate_review_gate(report, report_bytes, manifest)
            if manifest is not None else None)
    review_status: ReviewStatus = (
        "not_provided" if gate is None else gate["gate_status"]
    )
    ready = set(gate["comparison_ready_claim_ids"]) if gate is not None else set()
    # A blocked manifest cannot provide ANY positive review status, even
    # when some individual review decisions were filled in.
    if review_status != "review_complete":
        ready = set()

    source_map = {s.id: s for s in report.sources}
    evidence_map = {e.id: e for e in report.evidence}
    trigger_map: dict[str, set[str]] = {}
    for flagged in flag_claims_for_review(report).flags:
        trigger_map.setdefault(flagged.claim_id, set()).add(flagged.code)

    claims_by_slot: dict[tuple[str, str], list] = {}
    for claim in report.claims:
        claims_by_slot.setdefault((claim.company, claim.field), []).append(claim)

    rows: list[FieldRow] = []
    for company in report.request.companies:
        for field in FIELDS:
            slot = claims_by_slot.get((company, field), [])
            values: list[ClaimRow] = []
            for claim in slot:
                refs: list[EvidenceLink] = []
                for eid in claim.evidence_ids:
                    evidence = evidence_map[eid]
                    source = source_map[evidence.source_id]
                    refs.append(EvidenceLink(
                        evidence_id=eid,
                        source_id=source.id,
                        quote=evidence.quote,
                        source_title=source.title,
                        source_url=str(source.url),
                        retrieved_at=source.retrieved_at.isoformat(),
                    ))
                values.append(ClaimRow(
                    id=claim.id,
                    value=claim.value,
                    citations=refs,
                    review_trigger_codes=sorted(trigger_map.get(claim.id, set())),
                ))
            distinct = {_norm(x.value) for x in values}
            if not values:
                status: FieldStatus = "not_established"
                note = "No lexically accepted claim; evidence not established."
            elif len(distinct) > 1:
                status = "unresolved_conflict"
                note = (
                    "Different reported values; no source is automatically "
                    "preferred or interpreted as current."
                )
            elif review_status != "review_complete":
                status = "source_reported_unreviewed"
                note = "Verbatim excerpt linked; semantic support and field fit unreviewed."
            elif all(v.id in ready for v in values):
                status = "review_declared_ready"
                note = (
                    "Ready for comparison by user-supplied review decisions only; "
                    "not independent factuality validation or publication approval."
                )
            elif all(v.id not in ready for v in values):
                status = "review_declared_excluded"
                note = "User-supplied review decisions did not approve comparison."
            else:
                status = "mixed_review_decisions"
                note = "Inconsistent readiness within a field; do not summarize as verified."
            rows.append(FieldRow(
                company=company, field=field,
                status=status, values=values, note=note,
            ))

    counts = dict(sorted(Counter(row.status for row in rows).items()))
    return EvidencePreview(
        report_sha256=report_sha256(report_bytes),
        question=report.request.question,
        companies=list(report.request.companies),
        fields_per_company=len(FIELDS),
        comparison_rows=rows,
        status_counts=counts,
        source_reported_claims=len(report.claims),
        rejected_candidates=report.rejected_candidates,
        review_status=review_status,
        review_basis=(
            "no_review_manifest" if gate is None
            else "user_supplied_decisions_not_independently_authenticated"
        ),
    )


_STYLES = """
:root{color-scheme:light;font:15px system-ui,-apple-system,sans-serif}
body{max-width:1200px;margin:40px auto;padding:0 20px;color:#1f2937}
h1{font-size:1.6rem}h2{font-size:1.2rem}
.notice{background:#f3f4f6;border:1px solid #9ca3af;padding:14px;border-radius:8px}
table{width:100%;border-collapse:collapse;margin:24px 0;table-layout:fixed}
th,td{border-bottom:1px solid #d1d5db;vertical-align:top;padding:11px;text-align:left;overflow-wrap:anywhere}
th{background:#f3f4f6}td:first-child,th:first-child{width:13%}
td:nth-child(2),th:nth-child(2){width:16%}
td:nth-child(3),th:nth-child(3){width:19%}
code{background:#f3f4f6;padding:2px 4px}
.claim{margin:0 0 12px}p{line-height:1.45}
details{margin-top:5px;font-size:.9rem}summary{cursor:pointer}
small{color:#4b5563}
"""


def render_html(preview: EvidencePreview) -> str:
    """Static HTML, all source/model strings escaped. No scripts or active fetch."""
    esc = lambda text: html.escape(str(text), quote=True)
    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "<title>Company research — unverified evidence preview</title>",
        f"<style>{_STYLES}</style></head><body>",
        "<h1>Company intelligence — evidence preview (NOT FOR PUBLICATION)</h1>",
        "<div class=\"notice\"><strong>Publication blocked.</strong> ",
        esc(preview.warning), "</div>",
        f"<p><strong>Research question:</strong> {esc(preview.question)}</p>",
        f"<p><small>Report SHA-256: {esc(preview.report_sha256)}. ",
        f"Review: {esc(preview.review_status)}. Calculated observations: 0 ",
        "(not supported by current report schema).</small></p>",
        "<table><thead><tr><th>Company</th><th>Field</th>",
        "<th>Evidence status</th><th>Reported claims and citations</th>",
        "</tr></thead><tbody>",
    ]
    for row in preview.comparison_rows:
        parts.extend([
            "<tr><td>", esc(row.company), "</td><td>", esc(row.field),
            "</td><td><strong>", esc(row.status.replace("_", " ")),
            "</strong><p><small>", esc(row.note), "</small></p></td><td>",
        ])
        if not row.values:
            parts.append("<em>Not established</em>")
        for value in row.values:
            parts.extend([
                "<div class=\"claim\"><strong>", esc(value.id), ": </strong>",
                esc(value.value),
            ])
            if value.review_trigger_codes:
                parts.extend([
                    "<p><small>Review triggers: ",
                    esc(", ".join(value.review_trigger_codes)), "</small></p>"
                ])
            for evidence in value.citations:
                parts.extend([
                    "<details><summary>Evidence ", esc(evidence.evidence_id),
                    " — ", esc(evidence.source_title), "</summary><p>",
                    "<a href=\"", esc(evidence.source_url),
                    "\" rel=\"noopener noreferrer\" target=\"_blank\">",
                    esc(evidence.source_id), " source URL</a>",
                    " · retrieved ", esc(evidence.retrieved_at),
                    "<p><small>Publication date: unknown</small></p>",
                    "<blockquote>", esc(evidence.quote), "</blockquote>",
                    "</details>",
                ])
            parts.append("</div>")
        parts.append("</td></tr>")
    parts.extend([
        "</tbody></table><p><small>All claim values are source-reported. ",
        "A source quote containing the words of a claim does not establish ",
        "that the claim is correct. No financial metrics are computed from ",
        "unstructured statements.</small></p></body></html>\n",
    ])
    return "".join(parts)


def preview_files(
    report_bytes: bytes,
    output_dir: Path,
    manifest: ReviewManifest | None = None,
    *,
    existing_dir: bool = False,
) -> EvidencePreview:
    """Write two artifacts, refuse overwrite. Caller owns run directory lifecycle."""
    report = _report_from_exact_bytes(report_bytes)
    preview = build_preview(report, report_bytes, manifest)
    data = preview.model_dump(mode="json")
    json_bytes = (
        json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    html_bytes = render_html(preview).encode("utf-8")
    json_path = output_dir / "evidence_preview.json"
    html_path = output_dir / "evidence_preview.html"
    if existing_dir:
        if not output_dir.is_dir():
            raise ValueError("Existing run directory does not exist")
        if json_path.exists() or html_path.exists():
            raise FileExistsError("Cannot overwrite evidence preview")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    # Full validation happened before any write. Write atomically so an
    # interrupted write cannot leave malformed HTML or JSON.
    for path, payload in ((json_path, json_bytes), (html_path, html_bytes)):
        temp = path.with_name(path.name + ".tmp")
        temp.write_bytes(payload)
        temp.replace(path)
    return preview


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M2.7 offline company evidence preview")
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--review-labels", type=Path, default=None)
    args = p.parse_args(argv)
    raw = args.report.read_bytes()
    manifest = (ReviewManifest.model_validate_json(
        args.review_labels.read_bytes()
    ) if args.review_labels is not None else None)
    preview = preview_files(raw, args.output_dir, manifest)
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "report_sha256": preview.report_sha256,
        "review_status": preview.review_status,
        "comparison_slots": len(preview.comparison_rows),
        "source_reported_claims": preview.source_reported_claims,
        "status_counts": preview.status_counts,
        "calculated_observations": len(preview.calculated_observations),
        "publication_ready": False,
        "preview_html": str(args.output_dir / "evidence_preview.html"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
