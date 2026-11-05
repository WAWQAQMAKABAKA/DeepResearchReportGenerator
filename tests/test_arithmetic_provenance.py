"""M2.6.1 bounded arithmetic provenance contracts, with DISJOINT synthetic numbers."""
from __future__ import annotations

import json
from decimal import Decimal

import pytest

from deepresearch_v2.arithmetic_provenance import (
    SCHEMA_VERSION, audit_derived, classify_case, find_formula, main,
)
from deepresearch_v2.business_validation import audit_case, audit_existing_run
from deepresearch_v2.financebench_benchmark import (
    Case, Excerpt, FinanceAnswer, run,
)


def case(question, gold, passage, *, kind="metrics-generated", cid="new-synth-1"):
    return Case(
        id=cid, company="Fixture Only Inc",
        doc_name="FICTION_2024_10K", question_type=kind,
        question=question, gold_answer=gold,
        evidence=[Excerpt(id="E1", doc_name="FICTION_2024_10K", page=2,
                          text=passage)],
    )


def answer(response, rationale, ids=None):
    return FinanceAnswer(
        answer=response, rationale=rationale,
        evidence_ids=["E1"] if ids is None else ids,
    )


def classify(example, prediction):
    return classify_case(example, prediction, audit_case(example, prediction))


def test_direct_source_candidate_cannot_certify_semantics():
    c = case(
        "What is net plant value? Answer in USD billions.", "$12.30",
        "Source has 12,345 (in millions), unrelated line items include 1,345.",
    )
    m = answer("$12.345 billion", "The cited value was 12,345 million.")
    result = classify(c, m)
    assert result.provenance_kind == "numeric_literal_candidate"
    assert result.numeric_reference_status == "different"
    assert result.numeric_semantic_support == "unverified"
    assert result.metric_definition_correct == "unverified"
    assert not result.publication_ready


def test_revenue_yoy_on_synthetic_unseen_values():
    c = case(
        "What is the year-over-year change in revenue, in percent?",
        "20.0%",
        "Statements of operations. Net sales, year 2021 1,000, year 2022 1,200.",
    )
    a = answer(
        "20.0%", "Sales 2022=1,200, 2021=1,000; (1,200 − 1,000) / 1,000 = 20.0%"
    )
    result = classify(c, a)
    assert result.provenance_kind == "calculated_candidate"
    assert result.original_literal_status == "not_found"
    assert result.formula.formula == "revenue_yoy"
    assert result.formula.operands_literal_in_cited_evidence
    assert result.formula.arithmetic_relation == "consistent_at_answer_precision"
    assert Decimal(result.formula.expected_value) == Decimal("20")

    # A model explanation cannot fabricate source operands.
    c_missing = c.model_copy(update={
        "evidence": [Excerpt(id="E1", doc_name="FICTION_2024_10K", page=2,
                             text="Sales last year 1,000; current year unavailable.")]
    })
    missing = classify(c_missing, a)
    assert missing.provenance_kind == "unresolved_calculation"
    assert not missing.formula.operands_literal_in_cited_evidence

    # A convincing formula does NOT excuse an arithmetic mismatch.
    wrong = classify(c, answer("25.0%", a.rationale))
    assert wrong.provenance_kind == "unresolved_calculation"
    assert wrong.formula.arithmetic_relation == "disagrees_at_answer_precision"


def test_inventory_roughness_is_labeled_not_silently_corrected():
    c = case(
        "Roughly how many times has inventory sold? Calculate inventory turnover ratio.",
        "3.0 times",
        "Cost of products sold 900; inventories for two years: 200 400.",
    )
    a = answer(
        "3.01 times",
        "FY2024 cost of products sold = $900M. Average inventory = "
        "(200 + 400)/2 = 300. Turnover = 900 / 300 = 3.0 times.",
    )
    result = classify(c, a)
    assert result.provenance_kind == "calculated_candidate"
    assert result.formula.formula == "inventory_turnover"
    assert result.formula.arithmetic_relation == "approximate_with_roughly_qualifier"
    assert Decimal(result.formula.expected_value) == 3

    # Same answer without explicit roughness cannot pass at 2 decimal precision.
    c_precise = c.model_copy(update={
        "question": "Calculate inventory turnover ratio for the year.",
    })
    strict = classify(c_precise, a)
    assert strict.provenance_kind == "unresolved_calculation"
    assert strict.formula.arithmetic_relation == "disagrees_at_answer_precision"


def test_guidance_percentage_points_and_payout_ratio():
    guidance = case(
        "By how many percentage points did EPS guidance increase?",
        "1 percentage point",
        "Core constant currency EPS growth raised to 9 percent, previously 8 percent.",
    )
    result = classify(guidance, answer(
        "1 percentage point", "Guidance: 9% - 8% = 1 percentage point."
    ))
    assert result.provenance_kind == "calculated_candidate"
    assert result.formula.formula == "eps_guidance_delta"
    assert result.formula.arithmetic_relation == "consistent_at_answer_precision"

    payout = case(
        "What is the dividend payout ratio?", "0.50",
        "Cash dividends paid 50; income statement net income 100.",
    )
    verified = classify(payout, answer(
        "0.50", "Dividends paid = 50; net income = 100. Payout ratio = 50 / 100 = 0.50."
    ))
    assert verified.provenance_kind == "calculated_candidate"
    assert verified.formula.formula == "dividend_payout_ratio"
    assert verified.formula.arithmetic_relation == "consistent_at_answer_precision"

    missing_label = payout.model_copy(update={
        "evidence": [Excerpt(id="E1", text="Sales were 50, tax was 100.",
                             doc_name="FICTION_2024_10K", page=2)]
    })
    assert find_formula(missing_label, answer(
        "0.50", "Payout ratio = 50 / 100 = 0.50."
    )) is None


def test_unscaled_amount_not_mistaken_for_valid_currency_literal():
    c = case(
        "What are annual restructuring charges?", "$411 million",
        "Note 5 Restructuring charges: Total 411.",
    )
    m = answer("$411 million", "Line item gives 411.")
    result = classify(c, m)
    assert result.provenance_kind == "unresolved_scale"
    assert result.original_literal_status == "not_found"
    assert result.numeric_semantic_support == "unverified"


def test_no_source_or_compound_prose_is_ungraded():
    c = case(
        "What was the year-over-year change in revenue?", "20.0%",
        "Sales were 1,200 and 1,000.",
    )
    out = classify(c, answer(
        "20.0%", "(1,200 - 1,000) / 1,000 = 20%", ids=[]
    ))
    assert out.provenance_kind == "unresolved_calculation"
    assert not out.formula.operands_literal_in_cited_evidence

    multi = classify(c, answer(
        "Revenue rose 20.0% from 1,000 to 1,200.",
        "((1,200 - 1,000) / 1,000) = 20.0%.",
    ))
    assert multi.provenance_kind == "nonnumeric"
    assert not multi.publication_ready


def make_dataset(tmp_path):
    types = ("metrics-generated", "domain-relevant", "novel-generated")
    rows = []
    for n in range(12):
        rows.append({
            "financebench_id": f"fixture-disjoint-{n}",
            "company": "Fixture Inc",
            "doc_name": "FIXTURE_2024_10K",
            "question_type": types[n % 3],
            "question": f"What was the amount in year {n}?",
            "answer": "$200.00",
            "justification": "PRIVATE LABEL NOT A MODEL INPUT",
            "evidence": [{
                "evidence_text": "Balance sheet (in millions) asset 200, liabilities 100.",
                "doc_name": "FIXTURE_2024_10K",
                "evidence_page_num": 10,
            }],
        })
    path = tmp_path / "financebench.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


class Fake:
    def predict(self, item):
        return answer("$200.00", "Balance sheet amount 200 in millions."), {
            "input_tokens": 2, "output_tokens": 2, "total_tokens": 4
        }


def setup_prior(tmp_path):
    dataset = make_dataset(tmp_path)
    run_dir, prior = tmp_path / "run", tmp_path / "prior"
    run(dataset, run_dir, Fake(), count=6, model="mock")
    audit_existing_run(dataset, run_dir, prior)
    return dataset, run_dir, prior


def test_offline_end_to_end_sidecar_provenance_and_no_mutation(tmp_path, capsys):
    dataset, run_dir, prior = setup_prior(tmp_path)
    before = {x.name: x.read_bytes() for x in run_dir.iterdir() if x.is_file()}
    before_parent = (prior / "business_audit.json").read_bytes()
    out = tmp_path / "new_audit"
    result = audit_derived(dataset, run_dir, prior, out)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["sample_size"] == 6
    assert result["overall_business_accuracy"] is None
    assert result["publication_ready"] is False
    assert result["independently_human_validated"] is False
    assert result["provenance_kind_counts"] == {"numeric_literal_candidate": 6}
    assert len(result["cases"]) == 6
    assert result == json.loads((out / "arithmetic_provenance.json").read_text())
    assert {x.name: x.read_bytes() for x in run_dir.iterdir() if x.is_file()} == before
    assert (prior / "business_audit.json").read_bytes() == before_parent
    with pytest.raises(FileExistsError):
        audit_derived(dataset, run_dir, prior, out)

    # CLI should run without any provider API key, in a separate fresh directory.
    assert main([
        "--dataset", str(dataset), "--run-dir", str(run_dir),
        "--prior-audit-dir", str(prior),
        "--output-dir", str(tmp_path / "cli_result"),
    ]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["sample_size"] == 6
    assert cli["publication_ready"] is False


def test_fail_closed_on_tampering_parent_and_dataset(tmp_path):
    dataset, run_dir, prior = setup_prior(tmp_path)
    parent_path = prior / "business_audit.json"
    parent = json.loads(parent_path.read_text())
    parent["case_results"][0]["model_answer"] = "Tampered"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    out = tmp_path / "should_not_exist"
    with pytest.raises(ValueError, match="tampered"):
        audit_derived(dataset, run_dir, prior, out)
    assert not out.exists()
    # Restore by regenerating a distinct fresh parent, then mutate dataset.
    clean_prior = tmp_path / "fresh_parent"
    audit_existing_run(dataset, run_dir, clean_prior)
    dataset.write_bytes(dataset.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        audit_derived(dataset, run_dir, clean_prior, out)
    assert not out.exists()
