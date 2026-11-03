"""M2.1 adapter qualification. Network and real model calls are never made in tests."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from deepresearch_v2.core import Candidate, ExtractionBatch, ResearchRequest
from deepresearch_v2.live import LangChainExtractor, TavilyRetriever, company_queries
from deepresearch_v2.public_benchmark import SnapshotRetriever, load_snapshot
from deepresearch_v2.qualification import (
    ManualReview, _read_json, load_recorded_run, qualify_run, replay_run, review_run,
    score_manual_review,
)


class FakeTavilyClient:
    def __init__(self):
        self.calls = []

    def search(self, query, max_results, include_raw_content, topic):
        self.calls.append((query, max_results, include_raw_content, topic))
        company = next(c for c in ("Cursor", "GitHub Copilot", "Devin") if query.startswith(c + " "))
        slug = {"Cursor": "cursor", "GitHub Copilot": "copilot", "Devin": "devin"}[company]
        source = {
            "Cursor": "Cursor provides agent assistance in an editor for developer teams.",
            "GitHub Copilot": "GitHub Copilot supports coding agents using GitHub workflows.",
            "Devin": "Devin provides an AI software engineering agent for engineering teams.",
        }[company]
        return {"results": [
            {"url": f"https://docs.example.org/{slug}", "title": f"{company} page",
             "raw_content": source, "content": "fallback snippet"},
            {"url": "", "title": "bad", "content": "bad"},
        ]}


class FakeStructuredModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        user = messages[1]["content"]
        if "COMPANY: Cursor\n" in user:
            candidates = [
                {"company": "Cursor", "field": "product", "value": "Cursor's AI assistant",
                 "source_id": "S1", "quote": "Agent is Cursor's AI assistant."},
                {"company": "Cursor", "field": "commercial_traction", "value": "50 million subscribers",
                 "source_id": "S1", "quote": "Agent is Cursor's AI assistant."},
            ]
        elif "COMPANY: GitHub Copilot\n" in user:
            candidates = [
                {"company": "GitHub Copilot", "field": "product",
                 "value": "asynchronous, autonomous developer agent", "source_id": "S3",
                 "quote": "Copilot coding agent, our asynchronous, autonomous developer agent, is now generally available for all paid Copilot subscribers."},
            ]
        else:
            candidates = [
                {"company": "Devin", "field": "features", "value": "multiple parallel Devins",
                 "source_id": "S6", "quote": "Spin up multiple parallel Devins, each equipped with its own interactive, cloud-based IDE."},
            ]
        return {"parsed": {"items": candidates}, "raw": SimpleNamespace(content="model generated response"),
                "parsing_error": None}


class FailingExtractor:
    def extract(self, request, sources):
        raise RuntimeError("simulated model failure with SECRET_PLACEHOLDER")


def _fixture_components():
    snapshot = load_snapshot()
    model = FakeStructuredModel()
    extractor = LangChainExtractor(provider="test", model="mock-pydantic", structured_model=model)
    request = ResearchRequest.model_validate(snapshot["request"])
    return snapshot, request, SnapshotRetriever(snapshot), extractor, model


def test_bounded_tavily_adapter_injected_no_credentials(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    fake = FakeTavilyClient()
    request = ResearchRequest(question="Compare products", companies=["Cursor", "GitHub Copilot", "Devin"])
    provider = TavilyRetriever(client=fake, results_per_query=2)
    sources = provider.retrieve(request)
    assert len(sources) == 3
    assert len(fake.calls) == 6
    assert all(x[1:] == (2, True, "general") for x in fake.calls)
    assert len(provider.search_log) == 6
    assert [s.company for s in sources] == request.companies
    assert provider.source_provenance["S1"]["content_kind"] == "raw_content"
    assert provider.source_provenance["S1"]["truncated"] is False
    assert provider.source_provenance["S1"]["stored_chars"] > 0
    assert company_queries("Cursor")[0] in fake.calls[0][0]


def test_tavily_fails_closed_on_malformed_provider_response():
    class InvalidClient:
        def search(self, *args, **kwargs):
            return {"answer": "The source is missing"}
    provider = TavilyRetriever(client=InvalidClient())
    with pytest.raises(ValueError, match="malformed"):
        provider.retrieve(ResearchRequest(question="Compare products", companies=["A", "B", "C"]))


def test_live_model_contract_with_stub_and_source_capture(tmp_path):
    snap, request, retriever, extractor, model = _fixture_components()
    run_dir = tmp_path / "qualified"
    result = qualify_run(request, retriever, extractor, run_dir, engine="direct",
                         mode="llm-on-curated-excerpts", gold=snap)
    assert result["source_count"] == 8
    assert result["accepted_claim_count"] == 3
    assert result["rejected_candidate_count"] == 1
    assert result["gold_exact_alignment"]["matching_value_triplets"] == 3
    assert result["semantic_support_evaluated"] is False
    assert result["qualification_status"] == "UNREVIEWED_CLAIMS"
    assert len(model.calls) == 3
    assert all("UNTRUSTED" in call[0]["content"] for call in model.calls)
    assert all("TAVILY_API_KEY" not in p.read_text(encoding="utf-8") for p in run_dir.glob("*.json"))
    assert _read_json(run_dir / "status.json")["status"] == "completed_unreviewed"
    assert len(_read_json(run_dir / "candidates.json")["invocations"]) == 3
    replay_request, sources, batch = load_recorded_run(run_dir)
    assert replay_request == request and len(sources) == 8 and len(batch.items) == 4


def test_capture_replay_matches_stored_report(tmp_path):
    snap, request, retriever, extractor, _ = _fixture_components()
    initial = tmp_path / "initial"
    qualify_run(request, retriever, extractor, initial, engine="direct", gold=snap)
    later = tmp_path / "replayed"
    replay_run(initial, later, engine="direct")
    assert (initial / "report.json").read_bytes() == (later / "report.json").read_bytes()
    assert (initial / "report.md").read_bytes() == (later / "report.md").read_bytes()
    with pytest.raises(FileExistsError):
        replay_run(initial, later, engine="direct")


def test_replay_rejects_tampered_source_or_model_output(tmp_path):
    snap, request, retriever, extractor, _ = _fixture_components()
    run_dir = tmp_path / "recorded"
    qualify_run(request, retriever, extractor, run_dir, engine="direct", gold=snap)
    source_path = run_dir / "sources.json"
    data = _read_json(source_path)
    data["sources"][0]["text"] = "Tampered upstream document"
    source_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="source text has changed"):
        load_recorded_run(run_dir)


def test_model_failure_leaves_retrieval_snapshot_but_no_report(tmp_path):
    snap, request, retriever, extractor, _ = _fixture_components()
    run_dir = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated model failure"):
        qualify_run(request, retriever, FailingExtractor(), run_dir, engine="direct")
    assert (run_dir / "sources.json").exists()
    assert not (run_dir / "candidates.json").exists()
    assert not (run_dir / "report.json").exists()
    status = _read_json(run_dir / "status.json")
    assert status["status"] == "failed" and status["retrieval_captured"]
    assert "SECRET_PLACEHOLDER" not in (run_dir / "status.json").read_text()


def test_manual_evidence_review_is_separate_from_lexical_validation(tmp_path):
    snap, request, retriever, extractor, _ = _fixture_components()
    run_dir = tmp_path / "candidate"
    qualify_run(request, retriever, extractor, run_dir, engine="direct")
    report = _read_json(run_dir / "report.json")
    labels = {"reviewer": "synthetic-test-reviewer", "labels": [
        {"claim_id": "C1", "verdict": "supported", "note": "test label on synthetic model output"},
        {"claim_id": "C2", "verdict": "uncertain"},
    ]}
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps(labels), encoding="utf-8")
    review_output = tmp_path / "manual_review.json"
    review = review_run(run_dir / "report.json", label_path, review_output)
    assert review["review_status"] == "partial"
    assert review["unreviewed_claims"] == 1
    assert review["support_rate_among_reviewed"] == 0.5
    assert len(report["claims"]) == 3
    with pytest.raises(FileExistsError):
        review_run(run_dir / "report.json", label_path, review_output)
    bad = ManualReview.model_validate({"reviewer": "test", "labels": [
        {"claim_id": "C99", "verdict": "supported"}]})
    from deepresearch_v2.core import ResearchReport
    with pytest.raises(ValueError, match="Unknown claim"):
        score_manual_review(ResearchReport.model_validate(report), bad)
    with pytest.raises(ValidationError):
        ManualReview.model_validate({"reviewer": "test", "labels": [
            {"claim_id": "C1", "verdict": "supported"}, {"claim_id": "C1", "verdict": "stale"}]})


def test_live_mocked_graph_pipeline_if_langgraph_installed(tmp_path):
    pytest.importorskip("langgraph")
    snap, request, retriever, extractor, _ = _fixture_components()
    metrics = qualify_run(request, retriever, extractor, tmp_path / "graph", engine="graph", gold=snap)
    assert metrics["accepted_claim_count"] == 3
    metrics2 = replay_run(tmp_path / "graph", tmp_path / "replayed", engine="graph")
    assert metrics2["accepted_claim_count"] == 3
    assert (tmp_path / "graph" / "report.json").read_bytes() == (tmp_path / "replayed" / "report.json").read_bytes()


def test_oracle_cli_and_replay_using_real_world_excerpt_fixture(tmp_path):
    import subprocess
    import sys

    initial = tmp_path / "oracle"
    replayed = tmp_path / "replayed"
    subprocess.run([sys.executable, "-m", "deepresearch_v2.qualification", "oracle-fixture",
                    "--engine", "direct", "--output-dir", str(initial)],
                   capture_output=True, check=True, text=True)
    subprocess.run([sys.executable, "-m", "deepresearch_v2.qualification", "replay",
                    "--engine", "direct", "--run-dir", str(initial), "--output-dir", str(replayed)],
                   capture_output=True, check=True, text=True)
    assert (initial / "report.json").read_bytes() == (replayed / "report.json").read_bytes()
    assert _read_json(initial / "evaluation.json")["accepted_claim_count"] == 9
    assert _read_json(initial / "evaluation.json")["qualification_status"] == "UNREVIEWED_CLAIMS"
