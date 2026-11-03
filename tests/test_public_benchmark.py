"""M2 small real-world excerpt replay checks, not LLM accuracy tests."""
import json
import subprocess
import sys

import pytest

from deepresearch_v2.core import Candidate, ExtractionBatch, ResearchReport, compose_report
from deepresearch_v2.public_benchmark import (
    CuratedExtractor, SnapshotRetriever, evaluate, load_snapshot, run_public_snapshot,
)
from deepresearch_v2.core import ResearchRequest


def test_public_excerpts_replay_structured_and_traceable():
    snap = load_snapshot()
    report = run_public_snapshot("direct")
    assert len(report.sources) == 8
    assert len(report.claims) == 9
    assert report.rejected_candidates == 0
    assert report.model_validate_json(report.model_dump_json()) == report
    metrics = evaluate(report, snap)
    assert metrics["populated_company_fields"] == 9
    assert metrics["unpopulated_company_fields"] == 15
    assert metrics["extraction"] == "human_curated_not_llm"
    assert not metrics["semantic_support_assessed"]
    assert not metrics["live_model_assessed"]
    assert all(len(v) == 64 for v in metrics["source_digest_sha256"].values())


def test_missing_commercial_traction_stays_unknown():
    report = run_public_snapshot("direct")
    assert all(row["commercial_traction"] == "Not established" for row in report.components[0].rows)
    assert all(row["differentiation"] == "Not established" for row in report.components[0].rows)
    assert all(row["strengths"] == "Not established" for row in report.components[0].rows)


def test_stale_historical_devin_pricing_not_used_as_current():
    report = run_public_snapshot("direct")
    devin = report.components[0].rows[2]
    assert "Pro — $20/month" in devin["pricing"]
    assert "$500" not in devin["pricing"]
    assert any("$500" in s.text for s in report.sources)  # kept for audit, not promoted into pricing


def test_source_binding_and_invalid_quote_rejected():
    snap = load_snapshot()
    request = ResearchRequest.model_validate(snap["request"])
    sources = SnapshotRetriever(snap).retrieve(request)
    fabricated = Candidate(company="Devin", field="commercial_traction", value="100 million users",
                           source_id="S7", quote="Devin has 100 million users")
    bad_scope = Candidate(company="Cursor", field="pricing", value="$20/month",
                          source_id="S7", quote="Pro — $20/month, with included quota")
    report = compose_report(request, sources, ExtractionBatch(items=[fabricated, bad_scope]))
    assert report.rejected_candidates == 2
    assert not report.claims


def test_request_mismatch_refused():
    snap = load_snapshot()
    request = ResearchRequest(question="Another question", companies=snap["request"]["companies"])
    with pytest.raises(ValueError, match="Snapshot request"):
        SnapshotRetriever(snap).retrieve(request)


def test_public_cli_is_byte_stable(tmp_path):
    destination = tmp_path / "bench"
    cmd = [sys.executable, "-m", "deepresearch_v2.public_benchmark",
           "--engine", "direct", "--output-dir", str(destination)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    before = {name: (destination / name).read_bytes() for name in
              ("report.json", "report.md", "evaluation.json")}
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert before == {name: (destination / name).read_bytes() for name in before}
    assert ResearchReport.model_validate_json(before["report.json"]).profiles[0].company == "Cursor"
    assert json.loads(before["evaluation.json"])["source_count"] == 8


def test_public_snapshot_langgraph_matches_direct_when_available():
    pytest.importorskip("langgraph")
    assert run_public_snapshot("graph").model_dump_json() == run_public_snapshot("direct").model_dump_json()


def test_known_limitation_lexical_validator_not_semantic_entailment():
    # Demonstrates an actual failure mode: quote presence is not semantic support.
    # Do not interpret a passed lexical check as a verified claim.
    snap = load_snapshot()
    request = ResearchRequest.model_validate(snap["request"])
    source = SnapshotRetriever(snap).retrieve(request)[0]
    synthetic = source.model_copy(update={"text": "Cursor does not offer enterprise-only pricing."})
    misleading = Candidate(company="Cursor", field="pricing", value="enterprise-only pricing",
                            source_id=source.id, quote=synthetic.text)
    report = compose_report(request, [synthetic], ExtractionBatch(items=[misleading]))
    assert len(report.claims) == 1  # known false positive; must be fixed in evidence-quality milestone
