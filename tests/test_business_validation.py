"""M2.6 deterministic business QA audit contracts; no external data or model calls."""
from __future__ import annotations

import json
from decimal import Decimal

import pytest

from deepresearch_v2.business_validation import (
    SCHEMA_VERSION, audit_case, audit_existing_run, compare_numeric,
    parse_measurement,
)
from deepresearch_v2.financebench_benchmark import Case, Excerpt, FinanceAnswer, run


def example(*, question, gold, evidence_text, kind="domain-relevant"):
    return Case(
        id="synthetic-1",
        company="Synthetic Business Inc",
        doc_name="SYNTH_2023_10K",
        question_type=kind,
        question=question,
        gold_answer=gold,
        evidence=[
            Excerpt(id="E1", text=evidence_text, doc_name="SYNTH_2023_10K", page=3),
        ],
    )


def answer(text, rationale, ids=None):
    return FinanceAnswer(answer=text, rationale=rationale,
                         evidence_ids=["E1"] if ids is None else ids)


def codes(item):
    return {flag.code for flag in item.flags}


def test_numeric_units_precision_no_tolerance_inflation():
    q = "What was the annual net PP&E? Answer in USD billions."
    case = compare_numeric("$8.70", "$8.738 billion", q)
    assert case.status == "different"
    assert Decimal(case.absolute_difference) == Decimal("38000000.000")
    assert Decimal(case.reference_half_step) == Decimal("5000000.00")
    assert case.canonical_unit == "usd"

    nike = compare_numeric(
        "$16525.00", "16525", "How much current assets? Answer in USD millions."
    )
    assert nike.status == "exact"
    assert nike.predicted == nike.reference

    turnover = compare_numeric(
        "SampleCo sold inventory 2.7 times in FY2022.",
        "2.73 times", "Roughly how many times was inventory sold?"
    )
    assert turnover.status == "within_reference_precision"
    assert Decimal(turnover.absolute_difference) == Decimal("0.03")
    assert Decimal(turnover.reference_half_step) == Decimal("0.05")

    assert compare_numeric("0.8", "0.80", "What is the payout ratio?").status == "exact"


def test_compound_answer_and_currency_scale_fail_closed():
    question = "Did margins improve?"
    assert compare_numeric(
        "No. Gross margin decreased by 0.8%.",
        "No: margin was 18.5% versus 19.4%.",
        question,
    ).status == "not_comparable"
    assert compare_numeric(
        "$100 million", "$100 thousand", "How much?"
    ).status == "different"
    assert compare_numeric(
        "$10", "10", "How much?"
    ).status == "not_comparable"
    assert parse_measurement("No", question) is None
    # A million customers is NOT assumed to be millions of USD.
    assert parse_measurement("5 million users", question) is None
    one = parse_measurement(
        "Business guidance was raised by 1 percentage point.", question
    )
    assert one.amount == Decimal(1)
    assert one.unit == "percentage_points"
    assert parse_measurement("They could receive $66.56 per share.", question).unit == "usd_per_share"


def test_source_literal_is_diagnostic_not_semantic_validation():
    case = example(
        question="What was net PP&E? Answer in USD billions.",
        gold="$8.70",
        evidence_text="Balance sheet (in millions): PP&E net 8,738",
        kind="metrics-generated",
    )
    audit = audit_case(case, answer(
        "$8.738 billion", "Net property value 8,738 million from E1."
    ))
    assert audit.answer_numeric_literal_in_cited_evidence == "found"
    assert audit.numeric.status == "different"
    assert "numeric_reference_disagreement" in codes(audit)
    assert audit.semantic_correctness == "unverified"
    assert audit.publication_ready is False
    assert audit.cited_document_pages == ["SYNTH_2023_10K:p3"]


def test_industry_metric_proxy_context_and_statement_scope_are_warning_only():
    case = example(
        question="Does operating margin improve for this bank?",
        gold="Operating margin not an appropriate bank performance metric.",
        evidence_text="Statement of income: pretax income 9585, revenues 52862",
    )
    audit = audit_case(case, answer(
        "No, operating margin 18.1%", "Pretax income is the closest proxy available."
    ))
    assert "metric_definition_proxy" in codes(audit)
    assert audit.publication_ready is False

    liquid = example(
        question="Upon bankruptcy liquidation, what could each shareholder receive?",
        gold="$66.56 per share",
        evidence_text="Tangible book value per share was $66.56.",
    )
    check = audit_case(liquid, answer(
        "$66.56 per share", "TBVPS is a proxy for liquidation payout."
    ))
    assert check.numeric.status == "exact"
    assert "liquidation_book_value_assumption" in codes(check)
    assert check.publication_ready is False

    note = example(
        question="What restructuring is directly outlined in income statements?",
        gold="$411 million",
        evidence_text="Note 3 Restructuring and Impairment Charges: Total restructuring charges $411 million.",
    )
    check2 = audit_case(note, answer(
        "$411 million", "Note 3 shows the amount."
    ))
    assert "requested_statement_vs_note" in codes(check2)


def test_unknown_or_duplicate_citation_ids_fail_validation():
    case = example(question="Some question?", gold="10", evidence_text="Value 10")
    with pytest.raises(ValueError, match="Unknown evidence"):
        audit_case(case, answer("10", "From E7", ids=["E7"]))
    with pytest.raises(ValueError, match="Duplicate evidence"):
        audit_case(case, answer("10", "Duplicate", ids=["E1", "E1"]))


def make_dataset(tmp_path):
    rows = []
    types = ("metrics-generated", "domain-relevant", "novel-generated")
    for n in range(12):
        rows.append({
            "financebench_id": f"synthetic-{n}",
            "company": "Synthetic Company",
            "doc_name": "SYNTH_2023_10K",
            "question": f"What was the amount in year {n}?",
            "question_type": types[n % 3],
            "answer": "$100.00",
            "justification": "SECRET reference",
            "evidence": [{
                "evidence_text": "The financial report states amount 100 in USD millions.",
                "doc_name": "SYNTH_2023_10K",
                "evidence_page_num": 1,
            }],
        })
    file = tmp_path / "dataset.jsonl"
    file.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    return file


class Mock:
    def predict(self, item):
        return answer("$100.00", "Stated in E1"), {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4}


def test_offline_audit_provenance_and_immutable_sidecar(tmp_path):
    data = make_dataset(tmp_path)
    run_dir = tmp_path / "original_run"
    original = run(data, run_dir, Mock(), count=6, model="mock")
    snapshot = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    output = tmp_path / "audit"
    audit = audit_existing_run(data, run_dir, output)
    assert audit["schema_version"] == SCHEMA_VERSION
    assert audit["sample_size"] == 6
    assert audit["numeric_status_counts"] == {"exact": 6}
    assert audit["overall_business_accuracy"] is None
    assert audit["publication_ready"] is False
    assert audit["independently_human_validated"] is False
    assert len(audit["case_results"]) == 6
    assert original["sample_size"] == 6
    assert json.loads((output / "business_audit.json").read_text()) == audit
    assert snapshot == {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        audit_existing_run(data, run_dir, output)

    # A dataset mutation invalidates the captured artifact, even when parser
    # would still accept the source JSONL.
    data.write_bytes(data.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        audit_existing_run(data, run_dir, tmp_path / "audit_changed_data")


def test_tampered_evaluation_is_rejected_before_writing_sidecar(tmp_path):
    dataset = make_dataset(tmp_path)
    run_dir = tmp_path / "original"
    run(dataset, run_dir, Mock(), count=3)
    file = run_dir / "evaluation.json"
    contents = json.loads(file.read_text())
    contents["sample_size"] = 999
    file.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="does not match"):
        audit_existing_run(dataset, run_dir, tmp_path / "not_written")
    assert not (tmp_path / "not_written").exists()
