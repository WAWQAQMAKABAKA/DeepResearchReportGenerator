"""M2.3: conservative semantic/temporal review triggers over frozen evidence."""
from __future__ import annotations

import json
from pathlib import Path

from deepresearch_v2.core import Candidate, ExtractionBatch, ResearchRequest, compose_report
from deepresearch_v2.public_benchmark import CuratedExtractor, SnapshotRetriever, load_snapshot
from deepresearch_v2.qualification import qualify_run, replay_run
from deepresearch_v2.quality_flags import RULES_VERSION, flag_claims_for_review

_TRANSCRIPTION = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "m2p2_transcribed_candidates_2026-11-04.json"
)


def _observed_report():
    snapshot = load_snapshot()
    request = ResearchRequest.model_validate(snapshot["request"])
    sources = SnapshotRetriever(snapshot).retrieve(request)
    transcription = json.loads(_TRANSCRIPTION.read_text(encoding="utf-8"))
    assert transcription["fixture_kind"] == "user_terminal_transcription_not_raw_api_capture"
    batch = ExtractionBatch(items=[Candidate.model_validate(c)
                                   for c in transcription["candidates"]])
    return snapshot, transcription, request, sources, batch, compose_report(request, sources, batch)


def test_transcribed_live_17_claims_remain_lexically_accepted():
    _, _, _, _, batch, report = _observed_report()
    assert len(batch.items) == len(report.claims) == 17
    assert report.rejected_candidates == 0
    assert sum(len(p.claims_by_field) for p in report.profiles) == 10
    assert report.claims[15].id == "C16"
    assert report.claims[15].value == "$500 a month"


def test_review_triggers_reproduce_observed_risks_without_rejecting():
    _, _, _, _, _, report = _observed_report()
    diagnosis = flag_claims_for_review(report)
    assert diagnosis.rules_version == RULES_VERSION
    assert diagnosis.total_claims == 17
    assert diagnosis.flagged_claims == 7
    assert diagnosis.flag_count == 8
    assert diagnosis.by_code == {
        "eligibility_as_target_customer": 1,
        "historical_source_title": 2,
        "pricing_amount_without_tier": 4,
        "use_case_as_target_customer": 1,
    }
    per_claim = {}
    for flag in diagnosis.flags:
        per_claim.setdefault(flag.claim_id, set()).add(flag.code)
    assert per_claim == {
        "C6": {"pricing_amount_without_tier"},
        "C8": {"eligibility_as_target_customer"},
        "C9": {"pricing_amount_without_tier"},
        "C10": {"use_case_as_target_customer"},
        "C15": {"pricing_amount_without_tier"},
        "C16": {"pricing_amount_without_tier", "historical_source_title"},
        "C17": {"historical_source_title"},
    }
    assert diagnosis.assessment == "deterministic_review_triggers_not_verdicts"
    assert not hasattr(diagnosis, "semantic_support_evaluated")
    assert report.claims[15].id == "C16"  # Flagging does not mutate/reject.
    assert report.components[0].kind == "comparison_table"


def test_no_false_temporal_inference_from_retrieval_date_or_precise_price():
    _, _, request, sources, _, _ = _observed_report()
    source_s7 = next(s for s in sources if s.id == "S7")
    # Include tier in extracted value; the S7 quote supports it verbatim.
    precise = Candidate(company="Devin", field="pricing", value="Pro — $20/month",
                        source_id="S7", quote=source_s7.text)
    # Engineering teams is a segment, not an eligibility tagline.
    historical = Candidate(
        company="Devin", field="target_customers", value="engineering teams",
        source_id="S8", quote=next(s for s in sources if s.id == "S8").text,
    )
    report = compose_report(request, sources, ExtractionBatch(items=[precise, historical]))
    quality = flag_claims_for_review(report)
    assert report.rejected_candidates == 0
    assert [(f.claim_id, f.code) for f in quality.flags] == [
        ("C2", "historical_source_title")
    ]
    assert not any("retrieved_at" in f.basis for f in quality.flags)


def test_frozen_transcription_replays_without_model_and_preserves_flags(tmp_path):
    snapshot, transcription, request, _, _, _ = _observed_report()
    original, replayed = tmp_path / "m2p3-run", tmp_path / "m2p3-replay"
    metrics = qualify_run(
        request, SnapshotRetriever(snapshot), CuratedExtractor(transcription),
        original, mode="transcribed-study", engine="graph",
    )
    assert metrics["accepted_claim_count"] == 17
    assert metrics["rejected_candidate_count"] == 0
    assert metrics["claims_with_review_triggers"] == 7
    assert metrics["review_triggers_are_verdicts"] is False
    replay_metrics = replay_run(original, replayed, engine="graph")
    assert replay_metrics == metrics
    for name in (
        "report.json", "report.md", "candidate_diagnostics.json",
        "quality_flags.json", "evaluation.json",
    ):
        assert (original / name).read_bytes() == (replayed / name).read_bytes()


def test_oracle_remains_valid_with_review_flags(tmp_path):
    gold = load_snapshot()
    request = ResearchRequest.model_validate(gold["request"])
    path = tmp_path / "oracle"
    m = qualify_run(request, SnapshotRetriever(gold), CuratedExtractor(gold),
                    path, mode="oracle", engine="direct", gold=gold)
    assert m["accepted_claim_count"] == 9
    assert m["rejected_candidate_count"] == 0
    assert json.loads((path / "quality_flags.json").read_text())["assessment"] == (
        "deterministic_review_triggers_not_verdicts"
    )
