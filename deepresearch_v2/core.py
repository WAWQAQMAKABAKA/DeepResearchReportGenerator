"""Typed, fail-closed evidence pipeline. No UI, network, or model dependency."""
from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

FieldName = Literal[
    "product", "target_customers", "pricing", "features",
    "differentiation", "commercial_traction", "strengths", "limitations",
]
FIELDS: tuple[str, ...] = (
    "product", "target_customers", "pricing", "features",
    "differentiation", "commercial_traction", "strengths", "limitations",
)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3)
    companies: list[str] = Field(min_length=3, max_length=5)

    @field_validator("companies")
    @classmethod
    def distinct_companies(cls, companies: list[str]) -> list[str]:
        cleaned = [x.strip() for x in companies]
        if any(not x for x in cleaned) or len({x.casefold() for x in cleaned}) != len(cleaned):
            raise ValueError("Companies must be non-empty and unique")
        return cleaned


class Source(BaseModel):
    id: str
    company: str
    url: HttpUrl
    title: str
    text: str = Field(min_length=1)
    retrieved_at: datetime
    provider: str

    @field_validator("retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value


class Evidence(BaseModel):
    id: str
    source_id: str
    quote: str = Field(min_length=1)


class Candidate(BaseModel):
    """Untrusted model output; not evidence until checked against a Source."""
    company: str
    field: FieldName
    value: str = Field(min_length=1)
    source_id: str
    quote: str = Field(min_length=1)


class ExtractionBatch(BaseModel):
    items: list[Candidate]


class Claim(BaseModel):
    id: str
    company: str
    field: FieldName
    value: str
    evidence_ids: list[str] = Field(min_length=1)
    attribution: Literal["source_reported"] = "source_reported"


class CompanyProfile(BaseModel):
    company: str
    claims_by_field: dict[FieldName, list[str]] = Field(default_factory=dict)


class MetricObservation(BaseModel):
    """Reserved for later chart generation; not produced in M1."""
    company: str
    metric: str
    value: Decimal
    unit: str
    period: str
    evidence_ids: list[str] = Field(min_length=1)


class ReportComponent(BaseModel):
    kind: Literal["comparison_table", "timeline", "narrative", "chart"]
    title: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, str]] = Field(default_factory=list)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _value_span(text: str) -> str:
    """Conservative lexical check: factual value must occur in supporting quote."""
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


class ResearchReport(BaseModel):
    request: ResearchRequest
    sources: list[Source]
    evidence: list[Evidence]
    claims: list[Claim]
    profiles: list[CompanyProfile]
    components: list[ReportComponent]
    rejected_candidates: int = 0

    @model_validator(mode="after")
    def validate_references(self) -> "ResearchReport":
        sources = {s.id: s for s in self.sources}
        evidence = {e.id: e for e in self.evidence}
        claims = {c.id: c for c in self.claims}
        if len(sources) != len(self.sources) or len(evidence) != len(self.evidence) or len(claims) != len(self.claims):
            raise ValueError("Duplicate source, evidence, or claim IDs")
        for e in self.evidence:
            if e.source_id not in sources or _normalized(e.quote) not in _normalized(sources[e.source_id].text):
                raise ValueError(f"Evidence {e.id} lacks a verbatim source passage")
        for c in self.claims:
            for evidence_id in c.evidence_ids:
                if (evidence_id not in evidence or sources[evidence[evidence_id].source_id].company != c.company
                        or _value_span(c.value) not in _value_span(evidence[evidence_id].quote)):
                    raise ValueError(f"Claim {c.id} has invalid or cross-company evidence")
        if [p.company for p in self.profiles] != self.request.companies:
            raise ValueError("Profiles must match requested companies and order")
        for p in self.profiles:
            for field, ids in p.claims_by_field.items():
                if any(i not in claims or claims[i].company != p.company or claims[i].field != field for i in ids):
                    raise ValueError("Profile has an invalid claim reference")
        return self


class Retriever(Protocol):
    def retrieve(self, request: ResearchRequest) -> list[Source]: ...


class Extractor(Protocol):
    def extract(self, request: ResearchRequest, sources: list[Source]) -> ExtractionBatch: ...


def compose_report(request: ResearchRequest, sources: list[Source], batch: ExtractionBatch) -> ResearchReport:
    """Reject unverifiable candidates, retain conflicts, and compose a deterministic table."""
    source_map = {s.id: s for s in sources}
    company_map = {name.casefold(): name for name in request.companies}
    profiles = [CompanyProfile(company=name) for name in request.companies]
    by_company = {p.company: p for p in profiles}
    evidence: list[Evidence] = []
    claims: list[Claim] = []
    rejected = 0
    for item in batch.items:
        canonical_company = company_map.get(item.company.casefold())
        source = source_map.get(item.source_id)
        if (canonical_company is None or source is None or source.company != canonical_company
                or not item.value.strip() or _normalized(item.quote) not in _normalized(source.text)
                or _value_span(item.value) not in _value_span(item.quote)):
            rejected += 1
            continue
        eid, cid = f"E{len(evidence) + 1}", f"C{len(claims) + 1}"
        evidence.append(Evidence(id=eid, source_id=source.id, quote=item.quote))
        claim = Claim(id=cid, company=canonical_company, field=item.field,
                      value=item.value.strip(), evidence_ids=[eid])
        claims.append(claim)
        by_company[canonical_company].claims_by_field.setdefault(item.field, []).append(cid)

    claims_by_id = {c.id: c for c in claims}
    evidence_by_id = {e.id: e for e in evidence}
    rows: list[dict[str, str]] = []
    for profile in profiles:
        row = {"Company": profile.company}
        for field in FIELDS:
            linked = [claims_by_id[cid] for cid in profile.claims_by_field.get(field, [])]
            if not linked:
                row[field] = "Not established"
                continue
            distinct: dict[str, tuple[str, set[str]]] = {}
            for claim in linked:
                key = _normalized(claim.value).casefold()
                if key not in distinct:
                    distinct[key] = (claim.value, set())
                refs = distinct[key][1]
                refs.update(evidence_by_id[eid].source_id for eid in claim.evidence_ids)
            entries = []
            for value, refs in distinct.values():
                citations = " ".join(f"[{sid}]({source_map[sid].url})" for sid in sorted(refs))
                entries.append(f"{value} {citations}")
            row[field] = ("Multiple reports (unresolved): " if len(entries) > 1 else "") + "; ".join(entries)
        rows.append(row)
    component = ReportComponent(kind="comparison_table", title="Company comparison",
                                columns=["Company", *FIELDS], rows=rows)
    return ResearchReport(request=request, sources=sources, evidence=evidence, claims=claims,
                          profiles=profiles, components=[component], rejected_candidates=rejected)


def run_research(request: ResearchRequest, retriever: Retriever, extractor: Extractor) -> ResearchReport:
    """Dependency-injected offline/runtime path, independent of Streamlit and LangGraph."""
    sources = retriever.retrieve(request)
    batch = extractor.extract(request, sources)
    return compose_report(request, sources, batch)


def markdown_table(report: ResearchReport) -> str:
    """Deterministic Markdown projection; escape data but retain generated citation links."""
    table = report.components[0]
    def escape(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(table.columns) + " |"
    rule = "| " + " | ".join("---" for _ in table.columns) + " |"
    lines = [f"## {table.title}", "", header, rule]
    lines += ["| " + " | ".join(escape(row[col]) for col in table.columns) + " |" for row in table.rows]
    return "\n".join(lines)
