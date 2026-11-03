from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from deepresearch_v2.core import (
    Candidate, ExtractionBatch, ResearchRequest, Source,
    compose_report, markdown_table, run_research,
)
from deepresearch_v2.example import COMPANIES, QUESTION, FixtureExtractor, FixtureRetriever


@pytest.fixture
def research_request():
    return ResearchRequest(question=QUESTION, companies=COMPANIES)


def test_offline_end_to_end(research_request):
    report = run_research(research_request, FixtureRetriever(), FixtureExtractor())
    assert len(report.profiles) == 3
    assert len(report.claims) == 5
    assert report.rejected_candidates == 0
    assert len(report.components[0].rows) == 3
    assert "[S1](https://fixture.example/1)" in markdown_table(report)
    assert "Not established" in markdown_table(report)
    assert report.model_validate_json(report.model_dump_json()).profiles == report.profiles


def test_invalid_quote_is_rejected_not_fabricated(research_request):
    sources = FixtureRetriever().retrieve(research_request)
    batch = ExtractionBatch(items=[Candidate(company="AlphaCode", field="commercial_traction",
                                             value="1 million paying users", source_id="S1",
                                             quote="AlphaCode has 1 million paying users.")])
    report = compose_report(research_request, sources, batch)
    assert not report.claims and not report.evidence
    assert report.rejected_candidates == 1
    assert report.components[0].rows[0]["commercial_traction"] == "Not established"


def test_missing_sources_yield_missing_fields(research_request):
    report = compose_report(research_request, [], ExtractionBatch(items=[]))
    assert len(report.profiles) == 3
    assert all(value == "Not established" for row in report.components[0].rows
               for key, value in row.items() if key != "Company")


def test_conflicting_reported_values_preserved(research_request):
    sources = FixtureRetriever().retrieve(research_request)
    sources.append(Source(id="S4", company="AlphaCode", url="https://fixture.example/4",
                          title="Second synthetic excerpt", text="AlphaCode lists a paid-only plan.",
                          retrieved_at=datetime.now(timezone.utc), provider="fixture"))
    batch = ExtractionBatch(items=[
        Candidate(company="AlphaCode", field="pricing", value="Free tier", source_id="S1",
                  quote="AlphaCode lists a free tier."),
        Candidate(company="AlphaCode", field="pricing", value="Paid only", source_id="S4",
                  quote="AlphaCode lists a paid-only plan."),
    ])
    row = compose_report(research_request, sources, batch).components[0].rows[0]
    assert "Multiple reports (unresolved)" in row["pricing"]
    assert "[S1]" in row["pricing"] and "[S4]" in row["pricing"]


def test_cross_company_candidate_rejected(research_request):
    sources = FixtureRetriever().retrieve(research_request)
    batch = ExtractionBatch(items=[Candidate(company="BetaPilot", field="pricing", value="Free tier",
                                             source_id="S1", quote="AlphaCode lists a free tier.")])
    assert compose_report(research_request, sources, batch).rejected_candidates == 1


def test_genuine_quote_with_invented_value_is_rejected(research_request):
    sources = FixtureRetriever().retrieve(research_request)
    batch = ExtractionBatch(items=[Candidate(company="AlphaCode", field="commercial_traction",
                                             value="10 million users", source_id="S1",
                                             quote="AlphaCode lists a free tier.")])
    assert compose_report(research_request, sources, batch).rejected_candidates == 1


def test_request_rejects_invalid_company_lists():
    for names in (["Alpha", "Beta"], ["Alpha", "alpha", "Gamma"], ["", "Beta", "Gamma"]):
        with pytest.raises(ValidationError):
            ResearchRequest(question="Compare products", companies=names)


def test_report_rejects_tampered_evidence(research_request):
    report = run_research(research_request, FixtureRetriever(), FixtureExtractor())
    data = report.model_dump(mode="json")
    data["evidence"][0]["quote"] = "Completely invented quote"
    with pytest.raises(ValidationError):
        type(report).model_validate(data)


def test_report_rejects_cross_company_reference(research_request):
    report = run_research(research_request, FixtureRetriever(), FixtureExtractor())
    data = report.model_dump(mode="json")
    data["profiles"][1]["claims_by_field"] = {"pricing": ["C1"]}
    with pytest.raises(ValidationError):
        type(report).model_validate(data)


def test_typed_field_rejects_unknown_field():
    with pytest.raises(ValidationError):
        Candidate(company="AlphaCode", field="invented_revenue", value="42M", source_id="S1", quote="x")


def test_retrieval_failure_is_not_converted_to_success(research_request):
    class BrokenRetriever:
        def retrieve(self, request):
            raise RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        run_research(research_request, BrokenRetriever(), FixtureExtractor())


def test_optional_langgraph_adapter_if_installed(research_request):
    pytest.importorskip("langgraph")
    from deepresearch_v2.graph import build_graph
    result = build_graph(FixtureRetriever(), FixtureExtractor()).invoke({"request": research_request})
    assert result["report"].profiles[0].company == "AlphaCode"
