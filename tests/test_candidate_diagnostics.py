"""M2.2 forensic diagnosis and exact-span extraction contract regression."""
from __future__ import annotations

import json

from deepresearch_v2.core import (
    Candidate, ExtractionBatch, ResearchRequest, candidate_rejection_reason,
    compose_report,
)
from deepresearch_v2.live import PROMPT_VERSION, SYSTEM_PROMPT
from deepresearch_v2.public_benchmark import CuratedExtractor, SnapshotRetriever, load_snapshot
from deepresearch_v2.qualification import candidate_diagnostics, qualify_run, replay_run


def _pinned():
    gold = load_snapshot()
    request = ResearchRequest.model_validate(gold["request"])
    sources = SnapshotRetriever(gold).retrieve(request)
    return gold, request, sources


def _value_from_source(source):
    # First complete lexical token, guaranteed to be an exact passage substring.
    return next(token for token in source.text.split() if len(token) > 3).strip(".,:;()")


def test_rejection_reason_cases_stay_fail_closed():
    _, request, sources = _pinned()
    c = next(s for s in sources if s.company == "Cursor")
    d = next(s for s in sources if s.company == "Devin")
    valid = Candidate(company="Cursor", field="product",
                      value=_value_from_source(c), source_id=c.id, quote=c.text)
    assert candidate_rejection_reason(request, sources, valid) is None
    assert candidate_rejection_reason(request, sources, valid.model_copy(
        update={"company": "Cognition Devin"})) == "unknown_company"
    assert candidate_rejection_reason(request, sources, valid.model_copy(
        update={"source_id": "S999"})) == "unknown_source_id"
    assert candidate_rejection_reason(request, sources, valid.model_copy(
        update={"source_id": d.id})) == "source_company_mismatch"
    assert candidate_rejection_reason(request, sources, valid.model_copy(
        update={"quote": "This invented sentence is not in the captured page"})) == "quote_not_verbatim"
    assert candidate_rejection_reason(request, sources, valid.model_copy(
        update={"value": "A completely invented claim that does not appear in quote"})) == "value_not_in_quote"


def test_synthetic_failure_breakdown_matches_observed_categories():
    _, request, sources = _pinned()
    c = next(s for s in sources if s.company == "Cursor")
    g = next(s for s in sources if s.company == "GitHub Copilot")
    d = next(s for s in sources if s.company == "Devin")
    paraphrase = Candidate(company="Cursor", field="features", source_id=c.id,
                           quote=c.text, value="PARAPHRASED_FEATURE_MISSING_FROM_QUOTE")
    alias = Candidate(company="Cognition Devin", field="pricing", source_id=d.id,
                      quote=d.text, value="Devin")
    accepted = Candidate(company="GitHub Copilot", field="product", source_id=g.id,
                         quote=g.text, value=_value_from_source(g))
    batch = ExtractionBatch(items=[paraphrase] * 6 + [alias] * 4 + [accepted] * 2)
    diagnostics = candidate_diagnostics(request, sources, batch)
    assert diagnostics["rejection_breakdown"] == {
        "unknown_company": 4, "value_not_in_quote": 6
    }
    assert diagnostics["accepted_count"] == 2
    report = compose_report(request, sources, batch)
    assert report.rejected_candidates == 10
    assert len(report.claims) == 2


def test_prompt_explicitly_forbids_company_alias_and_paraphrase():
    assert PROMPT_VERSION == "company-extract-m2.2-exact-spans-v1"
    assert "Cognition Devin" in SYSTEM_PROMPT
    assert "CONTIGUOUS VERBATIM substring" in SYSTEM_PROMPT
    assert "OMIT the candidate" in SYSTEM_PROMPT


def test_oracle_diagnostics_and_replay_are_byte_identical(tmp_path):
    gold, request, _ = _pinned()
    a, b = tmp_path / "run", tmp_path / "replay"
    metrics = qualify_run(request, SnapshotRetriever(gold), CuratedExtractor(gold),
                          a, engine="direct", gold=gold)
    diagnostics = json.loads((a / "candidate_diagnostics.json").read_text())
    assert diagnostics["candidate_count"] == 9
    assert diagnostics["rejected_count"] == 0
    assert diagnostics["accepted_count"] == 9
    assert metrics["rejection_breakdown"] == {}
    replay_run(a, b, engine="direct")
    for filename in ("report.json", "report.md", "candidate_diagnostics.json"):
        assert (a / filename).read_bytes() == (b / filename).read_bytes()
