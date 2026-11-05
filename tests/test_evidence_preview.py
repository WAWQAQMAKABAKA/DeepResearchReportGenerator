"""M2.7 integrated company evidence preview: offline, no true human labeling."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from deepresearch_v2.core import (
    Candidate, ExtractionBatch, ResearchReport, ResearchRequest, Source,
    compose_report,
)
from deepresearch_v2.evidence_preview import (
    SCHEMA_VERSION, build_preview, main, preview_files, render_html,
)
from deepresearch_v2.public_benchmark import (
    CuratedExtractor, SnapshotRetriever, load_snapshot, run_public_snapshot,
)
from deepresearch_v2.qualification import (
    main as qualification_main, qualify_run, replay_run,
)
from deepresearch_v2.quality_flags import flag_claims_for_review
from deepresearch_v2.review_gate import (
    ReviewManifest, build_review_template,
)


def report_bytes(report):
    return (report.model_dump_json(indent=2) + "\n").encode("utf-8")


def _synth_report(*, conflicting: bool = True) -> ResearchReport:
    request = ResearchRequest(
        question="Compare the products of three entirely synthetic companies.",
        companies=["Example One", "Example Two", "Example Three"],
    )
    source = Source(
        id="S1", company="Example One", url="https://example.com/overview",
        title="Example <img src=x onerror=alert(1)>",
        text="The product is <script>alert(1)</script>. "
             "Another document says the product is beta.",
        retrieved_at=datetime(2026, 11, 3, tzinfo=timezone.utc),
        provider="synthetic-fixture",
    )
    candidates = [
        Candidate(company="Example One", field="product",
                  value="<script>alert(1)</script>", source_id="S1",
                  quote="The product is <script>alert(1)</script>."),
    ]
    if conflicting:
        candidates.append(Candidate(
            company="Example One", field="product",
            value="beta", source_id="S1",
            quote="Another document says the product is beta.",
        ))
    return compose_report(request, [source], ExtractionBatch(items=candidates))


def _all_declared_ready(report, raw):
    data = build_review_template(report, raw)
    data["reviewer"] = "synthetic-test-id-not-a-real-human-review"
    triggers = flag_claims_for_review(report)
    by_claim = {}
    for f in triggers.flags:
        by_claim.setdefault(f.claim_id, set()).add(f.code)
    for decision in data["decisions"]:
        decision.update(
            passage_support="supported",
            field_fit="appropriate",
            temporal_relevance="not_time_sensitive",
            ready_for_comparison=True,
            note="Synthetic test values only. No human fact checking.",
            addressed_trigger_codes=sorted(by_claim.get(decision["claim_id"], set())),
        )
    return data


def test_pinned_public_company_report_generates_all_slots_without_fabricated_support():
    report = run_public_snapshot("direct")
    raw = report_bytes(report)
    preview = build_preview(report, raw)
    assert preview.schema_version == SCHEMA_VERSION
    assert preview.companies == report.request.companies
    assert len(preview.comparison_rows) == 3 * 8
    assert preview.source_reported_claims == len(report.claims)
    assert preview.calculated_observations == []
    assert preview.calculation_status == "not_supported_by_current_report_schema"
    assert preview.review_status == "not_provided"
    assert preview.publication_ready is False
    assert preview.semantic_support_automatically_verified is False
    assert sum(preview.status_counts.values()) == len(preview.comparison_rows)
    assert any(row.status == "not_established" for row in preview.comparison_rows)
    assert all(
        item.source_reporting_status == "lexically_linked_semantically_unverified"
        for row in preview.comparison_rows for item in row.values
    )
    assert any(
        item.citations[0].source_url.startswith("https://")
        for row in preview.comparison_rows for item in row.values
    )


def test_conflicts_stay_unresolved_and_source_data_is_html_escaped():
    report = _synth_report()
    preview = build_preview(report, report_bytes(report))
    field = next(row for row in preview.comparison_rows
                 if row.company == "Example One" and row.field == "product")
    assert field.status == "unresolved_conflict"
    assert len(field.values) == 2
    assert any(row.status == "not_established" for row in preview.comparison_rows)
    html_doc = render_html(preview)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_doc
    assert "<script>alert(1)</script>" not in html_doc
    assert "&lt;img src=x onerror=alert(1)&gt;" in html_doc
    assert "<img src=x onerror=alert(1)>" not in html_doc
    assert "<script" not in html_doc
    assert "NOT FOR PUBLICATION" in html_doc
    assert "Publication blocked." in html_doc


def test_blank_and_partially_completed_manifest_never_promotes_review():
    report = _synth_report(conflicting=False)
    raw = report_bytes(report)
    blank = ReviewManifest.model_validate(build_review_template(report, raw))
    unreviewed = build_preview(report, raw, blank)
    assert unreviewed.review_status == "blocked"
    assert "review_declared_ready" not in unreviewed.status_counts
    data = build_review_template(report, raw)
    data["reviewer"] = "synthetic-but-incomplete"
    data["decisions"][0]["passage_support"] = "supported"
    incomplete = build_preview(report, raw, ReviewManifest.model_validate(data))
    assert incomplete.review_status == "blocked"
    assert "review_declared_ready" not in incomplete.status_counts
    assert incomplete.publication_ready is False


def test_synthetic_complete_review_is_declared_not_independent_truth():
    report = _synth_report(conflicting=False)
    raw = report_bytes(report)
    ready = ReviewManifest.model_validate(_all_declared_ready(report, raw))
    preview = build_preview(report, raw, ready)
    product = next(row for row in preview.comparison_rows
                   if row.company == "Example One" and row.field == "product")
    assert preview.review_status == "review_complete"
    assert product.status == "review_declared_ready"
    assert preview.publication_ready is False
    assert preview.semantic_support_automatically_verified is False
    assert preview.review_basis == "user_supplied_decisions_not_independently_authenticated"

    labels = _all_declared_ready(report, raw)
    labels["decisions"][0]["ready_for_comparison"] = False
    excluded = build_preview(report, raw, ReviewManifest.model_validate(labels))
    product_excluded = next(row for row in excluded.comparison_rows
                            if row.company == "Example One" and row.field == "product")
    assert product_excluded.status == "review_declared_excluded"
    assert excluded.publication_ready is False


def test_even_declared_review_cannot_erase_conflicting_values():
    report = _synth_report(conflicting=True)
    raw = report_bytes(report)
    labels = ReviewManifest.model_validate(_all_declared_ready(report, raw))
    preview = build_preview(report, raw, labels)
    product = next(row for row in preview.comparison_rows
                   if row.company == "Example One" and row.field == "product")
    assert preview.review_status == "review_complete"
    assert product.status == "unresolved_conflict"
    assert preview.publication_ready is False


def test_exact_report_sha_enforced_before_writing_and_no_overwrite(tmp_path):
    report = _synth_report(conflicting=False)
    raw = report_bytes(report)
    manifest = ReviewManifest.model_validate(_all_declared_ready(report, raw))
    altered = raw + b" "
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="SHA-256"):
        preview_files(altered, output, manifest)
    assert not output.exists()

    first = preview_files(raw, output)
    assert first.review_status == "not_provided"
    assert json.loads((output / "evidence_preview.json").read_text())["report_sha256"] == first.report_sha256
    with pytest.raises(FileExistsError):
        preview_files(raw, output)
    with pytest.raises(FileExistsError):
        preview_files(raw, output, existing_dir=True)


def test_pipeline_generates_preview_in_real_headless_capture_and_replay(tmp_path):
    gold = load_snapshot()
    request = ResearchRequest.model_validate(gold["request"])
    run_dir = tmp_path / "live"
    metrics = qualify_run(
        request, SnapshotRetriever(gold), CuratedExtractor(gold), run_dir,
        engine="direct", mode="human-curated-oracle", gold=gold,
    )
    assert metrics["accepted_claims"] > 0
    saved_preview = json.loads((run_dir / "evidence_preview.json").read_text())
    assert saved_preview["publication_ready"] is False
    assert saved_preview["review_status"] == "not_provided"
    assert saved_preview["report_sha256"]
    assert (run_dir / "evidence_preview.html").exists()
    assert (run_dir / "status.json").exists()

    replay_dir = tmp_path / "replay"
    replay_run(run_dir, replay_dir, engine="graph")
    replayed = json.loads((replay_dir / "evidence_preview.json").read_text())
    assert replayed == saved_preview
    assert (run_dir / "report.json").read_bytes() == (replay_dir / "report.json").read_bytes()
    assert (run_dir / "report.md").read_bytes() == (replay_dir / "report.md").read_bytes()


def test_qualification_cli_supports_post_run_preview_without_model(tmp_path, capsys):
    report = _synth_report(conflicting=False)
    path = tmp_path / "report.json"
    path.write_bytes(report_bytes(report))
    dest = tmp_path / "preview"
    assert qualification_main([
        "report-preview", "--report", str(path), "--output-dir", str(dest)
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["review_status"] == "not_provided"
    assert result["comparison_slots"] == 24
    assert result["publication_ready"] is False
    assert (dest / "evidence_preview.html").exists()
    with pytest.raises(FileExistsError):
        main(["--report", str(path), "--output-dir", str(dest)])
