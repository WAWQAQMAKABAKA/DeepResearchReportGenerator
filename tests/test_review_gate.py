"""M2.4 human review packet and fail-closed readiness checks (offline)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepresearch_v2.core import Candidate, ExtractionBatch, ResearchRequest, compose_report
from deepresearch_v2.public_benchmark import SnapshotRetriever, load_snapshot
from deepresearch_v2.qualification import assess_review_gate, prepare_review_packet
from deepresearch_v2.review_gate import ReviewManifest, evaluate_review_gate


def _test_report(tmp_path):
    gold = load_snapshot()
    request = ResearchRequest.model_validate(gold["request"])
    sources = SnapshotRetriever(gold).retrieve(request)
    transcribed_path = (Path(__file__).resolve().parent.parent / "fixtures"
                        / "m2p2_transcribed_candidates_2026-11-04.json")
    transcribed = json.loads(transcribed_path.read_text())
    batch = ExtractionBatch(items=[
        Candidate.model_validate(x) for x in transcribed["candidates"]
    ])
    report = compose_report(request, sources, batch)
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _complete_decisions(template):
    template["reviewer"] = "synthetic-reviewer-for-unit-test"
    for d in template["decisions"]:
        d.update(
            passage_support="supported",
            field_fit="appropriate",
            temporal_relevance="not_time_sensitive",
            ready_for_comparison=True,
            note="This is a synthetic test decision, not a true human review.",
        )
    return template


def test_packet_exposes_17_claims_and_no_fabricated_review(tmp_path):
    path = _test_report(tmp_path)
    packet_path, blank_path = tmp_path / "packet.json", tmp_path / "template.json"
    result = prepare_review_packet(path, packet_path, blank_path)
    packet = json.loads(packet_path.read_text())
    template = json.loads(blank_path.read_text())
    assert result["packet_status"] == "unreviewed"
    assert packet["claims_total"] == 17
    assert packet["claims"][15]["claim_id"] == "C16"
    assert packet["claims"][15]["evidence"][0]["publication_date"] is None
    assert packet["claims"][15]["evidence"][0]["quote"].startswith("Today")
    assert {f["code"] for f in packet["claims"][15]["review_triggers"]} == {
        "historical_source_title", "pricing_amount_without_tier"
    }
    assert template["reviewer"] == ""
    assert all(d["passage_support"] is None for d in template["decisions"])
    assert all(d["ready_for_comparison"] is None for d in template["decisions"])
    assert packet["report_sha256"] == template["report_sha256"]
    with pytest.raises(FileExistsError):
        prepare_review_packet(path, packet_path, blank_path)


def test_review_gate_blocks_unreviewed_template(tmp_path):
    path = _test_report(tmp_path)
    packet, template_path = tmp_path / "p.json", tmp_path / "t.json"
    prepare_review_packet(path, packet, template_path)
    out = tmp_path / "gate.json"
    result = assess_review_gate(path, template_path, out)
    assert result["gate_status"] == "blocked"
    assert result["comparison_ready_claim_ids"] == []
    assert result["publication_generated"] is False
    assert result["decisions_supplied"] == 17
    assert len(result["blockers"]) == 18  # all pending, plus reviewer missing
    with pytest.raises(FileExistsError):
        assess_review_gate(path, template_path, out)


def test_supported_is_not_enough_to_publish_and_flag_acknowledgement_required(tmp_path):
    path = _test_report(tmp_path)
    pack, blank = tmp_path / "packet.json", tmp_path / "template.json"
    prepare_review_packet(path, pack, blank)
    packet = json.loads(pack.read_text())
    decision_input = _complete_decisions(json.loads(blank.read_text()))
    # First: all reviewers claim "supported", but none address existing flags.
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps(decision_input))
    blocked = assess_review_gate(path, labels, tmp_path / "blocked.json")
    assert blocked["gate_status"] == "blocked"
    assert blocked["comparison_ready_count"] == 0
    assert sorted({x["reason"] for x in blocked["blockers"]}) == [
        "unaddressed_review_triggers"
    ]

    flagged = {item["claim_id"]: {t["code"] for t in item["review_triggers"]}
               for item in packet["claims"]}
    for d in decision_input["decisions"]:
        d["addressed_trigger_codes"] = sorted(flagged[d["claim_id"]])
        if d["claim_id"] == "C16":
            d["temporal_relevance"] = "historical"
            d["ready_for_comparison"] = False
        if d["claim_id"] in {"C8", "C10"}:
            d["field_fit"] = "inappropriate"
            d["ready_for_comparison"] = False
    labels.write_text(json.dumps(decision_input))
    gate = assess_review_gate(path, labels, tmp_path / "reviewed.json")
    assert gate["gate_status"] == "review_complete"
    assert gate["comparison_ready_count"] == 14
    assert "C16" not in gate["comparison_ready_claim_ids"]
    assert "C8" not in gate["comparison_ready_claim_ids"]
    assert "C10" not in gate["comparison_ready_claim_ids"]
    assert gate["publication_generated"] is False
    assert gate["review_basis"] == "user_supplied_decisions_not_independently_authenticated"


def test_ready_requires_source_support_field_fit_and_temporal_relevance(tmp_path):
    path = _test_report(tmp_path)
    packet, blank = tmp_path / "p.json", tmp_path / "t.json"
    prepare_review_packet(path, packet, blank)
    packet_data = json.loads(packet.read_text())
    doc = _complete_decisions(json.loads(blank.read_text()))
    for d in doc["decisions"]:
        d["addressed_trigger_codes"] = sorted(
            f["code"] for x in packet_data["claims"] if x["claim_id"] == d["claim_id"]
            for f in x["review_triggers"]
        )
    for key, val in [
        ("passage_support", "insufficient"),
        ("field_fit", "inappropriate"),
        ("temporal_relevance", "historical"),
    ]:
        modified = json.loads(json.dumps(doc))
        modified["decisions"][0][key] = val
        labels = ReviewManifest.model_validate(modified)
        gate = evaluate_review_gate(
            __import__("deepresearch_v2.core", fromlist=["ResearchReport"]).ResearchReport.model_validate_json(path.read_bytes()),
            path.read_bytes(), labels
        )
        assert gate["gate_status"] == "blocked"
        assert any(x["reason"] == "invalid_ready_decision" for x in gate["blockers"])


def test_review_sha_binding_unknown_ids_and_duplicate_are_fail_closed(tmp_path):
    path = _test_report(tmp_path)
    p, t = tmp_path / "p.json", tmp_path / "t.json"
    prepare_review_packet(path, p, t)
    data = json.loads(t.read_text())
    report = __import__("deepresearch_v2.core", fromlist=["ResearchReport"]).ResearchReport.model_validate_json(path.read_bytes())
    altered = json.loads(json.dumps(data))
    altered["report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_review_gate(report, path.read_bytes(), ReviewManifest.model_validate(altered))
    altered = json.loads(json.dumps(data))
    altered["decisions"][0]["claim_id"] = "unknown"
    with pytest.raises(ValueError, match="Unknown reviewed claim"):
        evaluate_review_gate(report, path.read_bytes(), ReviewManifest.model_validate(altered))
    altered = json.loads(json.dumps(data))
    altered["decisions"][0]["claim_id"] = altered["decisions"][1]["claim_id"]
    with pytest.raises(ValueError, match="Duplicate claim IDs"):
        ReviewManifest.model_validate(altered)


def test_stale_and_unfit_claims_can_be_explicitly_excluded_without_false_approval(tmp_path):
    path = _test_report(tmp_path)
    p, t = tmp_path / "p.json", tmp_path / "t.json"
    prepare_review_packet(path, p, t)
    manifest = _complete_decisions(json.loads(t.read_text()))
    for d in manifest["decisions"]:
        d["ready_for_comparison"] = False
    outcome = evaluate_review_gate(
        __import__("deepresearch_v2.core", fromlist=["ResearchReport"]).ResearchReport.model_validate_json(path.read_bytes()),
        path.read_bytes(), ReviewManifest.model_validate(manifest)
    )
    assert outcome["gate_status"] == "review_complete"
    assert outcome["comparison_ready_count"] == 0
    assert outcome["publication_generated"] is False
