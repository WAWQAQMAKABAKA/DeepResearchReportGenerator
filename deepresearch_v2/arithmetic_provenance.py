"""M2.6.1 bounded arithmetic provenance over a frozen M2.6 FinanceBench audit.

Detect *candidate* literals or arithmetic patterns; never assert entailment.
No model calls, no arbitrary code execution, no changes to prior artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, DivisionByZero, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .business_validation import (
    SCHEMA_VERSION as PARENT_SCHEMA,
    AuditCase,
    audit_case,
    parse_measurement,
    source_literal_check,
)
from .financebench_benchmark import (
    Case, Excerpt, FinanceAnswer, load_data, replay_score, sample,
)

SCHEMA_VERSION = "m2.6.1-arithmetic-provenance-v1"

ProvenanceKind = Literal[
    "numeric_literal_candidate", "calculated_candidate", "unresolved_scale",
    "unresolved_calculation", "unresolved", "nonnumeric",
]
ArithmeticRelation = Literal[
    "consistent_at_answer_precision", "approximate_with_roughly_qualifier",
    "disagrees_at_answer_precision", "not_checked",
]


class FormulaCheck(BaseModel):
    formula: Literal[
        "revenue_yoy", "inventory_turnover", "eps_guidance_delta",
        "dividend_payout_ratio",
    ]
    operand_values: list[str]
    operands_literal_in_cited_evidence: bool
    expected_value: str | None
    predicted_value: str | None
    unit: str | None
    arithmetic_relation: ArithmeticRelation
    notes: str


class ProvenanceCase(BaseModel):
    id: str
    numeric_reference_status: str
    original_literal_status: str
    provenance_kind: ProvenanceKind
    formula: FormulaCheck | None
    advisory_flags_from_m2p6: list[str]
    numeric_semantic_support: Literal["unverified"] = "unverified"
    metric_definition_correct: Literal["unverified"] = "unverified"
    publication_ready: Literal[False] = False
    explanation: str = Field(min_length=1)


NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
NUM_PATTERN = re.compile(rf"(?<![\w])(?P<num>{NUM})(?![\w])")
REVENUE = re.compile(
    rf"\(\s*(?P<current>{NUM})\s*[−–-]\s*(?P<prior>{NUM})\s*\)\s*/\s*"
    rf"(?P<denom>{NUM})",
)
INVENTORY_AVERAGE = re.compile(
    rf"average\s+inventor(?:y|ies)\s*=\s*\(\s*(?P<first>{NUM})\s*\+\s*"
    rf"(?P<second>{NUM})\s*\)\s*/\s*2",
    re.IGNORECASE,
)
INVENTORY_COST = re.compile(
    rf"cost\s+of\s+products\s+sold\s*=\s*\$?\s*(?P<cost>{NUM})",
    re.IGNORECASE,
)
GUIDANCE = re.compile(
    rf"(?P<new>{NUM})\s*%\s*[−–-]\s*(?P<old>{NUM})\s*%\s*=\s*"
    rf"(?P<delta>{NUM})\s*percentage\s+points?\b",
    re.IGNORECASE,
)
DIVIDENDS = re.compile(
    rf"payout\s+ratio\s*=\s*(?P<dividend>{NUM})\s*/\s*(?P<net_income>{NUM})",
    re.IGNORECASE,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(raw: str) -> Decimal:
    return Decimal(raw.replace(",", ""))


def _cited(case: Case, prediction: FinanceAnswer) -> list[Excerpt]:
    allowed = {e.id for e in case.evidence}
    if not set(prediction.evidence_ids) <= allowed:
        raise ValueError(f"Unknown cited evidence for {case.id}")
    return [e for e in case.evidence if e.id in prediction.evidence_ids]


def _source_values(excerpts: list[Excerpt]) -> set[Decimal]:
    return {
        _number(m.group("num"))
        for passage in excerpts
        for m in NUM_PATTERN.finditer(passage.text)
    }


def _single_number_measurement(answer: str, question: str):
    """Conservative: never grade an answer with extra numbers as one scalar."""
    if len(list(NUM_PATTERN.finditer(answer))) != 1:
        return None
    return parse_measurement(answer, question)


def _make_formula(
    name: FormulaCheck.model_fields["formula"].annotation,
    operands: list[Decimal],
    expected: Decimal,
    expected_unit: str,
    case: Case,
    prediction: FinanceAnswer,
    notes: str,
) -> FormulaCheck:
    cited = _cited(case, prediction)
    numbers = _source_values(cited)
    present = bool(cited) and all(value in numbers for value in operands)
    reported = _single_number_measurement(prediction.answer, case.question)
    reported_value = reported.amount if reported is not None else None
    relation: ArithmeticRelation = "not_checked"
    if present and reported is not None and reported.unit == expected_unit:
        difference = abs(reported.amount - expected)
        half_step = reported.resolution / Decimal(2)
        if difference <= half_step:
            relation = "consistent_at_answer_precision"
        elif ("roughly" in case.question.casefold() or "approximately" in case.question.casefold()) and (
            expected != 0 and difference / abs(expected) <= Decimal("0.01")
        ):
            # Explicit roughness is only advisory and does not silently widen
            # normal rounding tolerance.
            relation = "approximate_with_roughly_qualifier"
        else:
            relation = "disagrees_at_answer_precision"
    return FormulaCheck(
        formula=name,
        operand_values=[str(x) for x in operands],
        operands_literal_in_cited_evidence=present,
        expected_value=str(expected),
        predicted_value=str(reported_value) if reported_value is not None else None,
        unit=expected_unit,
        arithmetic_relation=relation,
        notes=notes,
    )


def find_formula(case: Case, prediction: FinanceAnswer) -> FormulaCheck | None:
    """Parse exactly one whitelisted arithmetic shape from model rationale.

    Question + formula shape + cited source operand presence must ALL hold for
    a positive relation. A rationale alone is not trusted as a source.
    """
    q = case.question.casefold()
    text = prediction.rationale
    try:
        if ("revenue" in q or "net sales" in q) and (
            "year-over-year" in q or "year over year" in q
        ):
            m = REVENUE.search(text)
            if m:
                current, prior, denominator = (
                    _number(m.group(k)) for k in ("current", "prior", "denom")
                )
                if prior != denominator:
                    return None
                return _make_formula(
                    "revenue_yoy", [current, prior],
                    (current - prior) / prior * Decimal(100), "percent",
                    case, prediction,
                    "YOY arithmetic and source operand occurrence only; "
                    "correct year/metric column association is not proven.",
                )
        if "inventory turnover" in q or (
            "sold its inventory" in q and "times" in q
        ):
            a = INVENTORY_AVERAGE.search(text)
            c = INVENTORY_COST.search(text)
            if a and c:
                first = _number(a.group("first"))
                second = _number(a.group("second"))
                cost = _number(c.group("cost"))
                return _make_formula(
                    "inventory_turnover", [cost, first, second],
                    cost / ((first + second) / Decimal(2)), "times",
                    case, prediction,
                    "COGS / arithmetic-average inventory; source operand "
                    "occurrence does not validate inventory accounting scope.",
                )
        if "guidance" in q and "eps" in q and "percentage point" in q:
            m = GUIDANCE.search(text)
            if m:
                new, old, claimed = (
                    _number(m.group(k)) for k in ("new", "old", "delta")
                )
                if new - old != claimed:
                    return None
                return _make_formula(
                    "eps_guidance_delta", [new, old],
                    new - old, "percentage_points", case, prediction,
                    "Percentage-point subtraction only; historical guidance "
                    "version and period alignment are not proven.",
                )
        if "dividend payout ratio" in q:
            m = DIVIDENDS.search(text)
            if m:
                div = _number(m.group("dividend"))
                income = _number(m.group("net_income"))
                excerpts = _cited(case, prediction)
                content = " ".join(x.text.casefold() for x in excerpts)
                if "dividends" not in content or "net income" not in content:
                    return None
                return _make_formula(
                    "dividend_payout_ratio", [div, income],
                    div / income, "unitless", case, prediction,
                    "Dividends / net income; source operand occurrence does "
                    "not establish attribution or reporting period.",
                )
    except (DivisionByZero, InvalidOperation, ZeroDivisionError):
        # Never invent a result from zero or invalid denominators.
        return None
    return None


def _unscaled_amount_appears(
    case: Case, prediction: FinanceAnswer
) -> bool:
    """Detect a likely scale mismatch without claiming currency-unit support."""
    value = parse_measurement(prediction.answer, case.question)
    if value is None or value.unit != "usd":
        return False
    numbers = _source_values(_cited(case, prediction))
    # A filing note may say just "411" under an omitted "(in millions)" header.
    return (
        value.amount / (Decimal(10) ** 6) in numbers
        or value.amount / (Decimal(10) ** 9) in numbers
    )


def classify_case(
    case: Case, prediction: FinanceAnswer, parent: AuditCase
) -> ProvenanceCase:
    if case.id != parent.id or prediction.answer != parent.model_answer:
        raise ValueError("Parent audit does not match source case and prediction")
    measure = _single_number_measurement(prediction.answer, case.question)
    # Preserve parent-v1 literal semantics for integrity checking. V2's
    # separate conservative scalar reader must not rewrite history.
    legacy_measure = parse_measurement(prediction.answer, case.question)
    literal = source_literal_check(prediction, case, legacy_measure)
    if literal != parent.answer_numeric_literal_in_cited_evidence:
        raise ValueError("Parent audit literal diagnostic mismatch")
    formula = find_formula(case, prediction)
    if measure is None:
        kind: ProvenanceKind = "nonnumeric"
        explanation = (
            "Answer has no unambiguous single numeric measurement; "
            "semantic meaning remains unresolved."
        )
    elif literal == "found":
        kind = "numeric_literal_candidate"
        explanation = (
            "Matching numeric token (possibly with scale normalization) "
            "occurs in cited excerpt; row, year and metric entailment are unverified."
        )
    elif formula is not None and formula.operands_literal_in_cited_evidence and (
        formula.arithmetic_relation in {
            "consistent_at_answer_precision", "approximate_with_roughly_qualifier"
        }
    ):
        kind = "calculated_candidate"
        explanation = (
            "Bounded whitelisted formula matches numeric operands occurring in "
            "cited excerpts and approximately reproduces output; operand labels, "
            "source period and financial meaning remain unverified."
        )
    elif formula is not None:
        kind = "unresolved_calculation"
        explanation = (
            "Recognized formula but source operands were missing, model output "
            "did not match arithmetic, or unit/answer format was incomparable."
        )
    elif _unscaled_amount_appears(case, prediction):
        kind = "unresolved_scale"
        explanation = (
            "Unscaled numeric token appears in cited excerpt but the currency "
            "magnitude/unit cannot be proven from current evidence parsing."
        )
    else:
        kind = "unresolved"
        explanation = (
            "No qualified direct numeric token or whitelisted calculation "
            "established from citations; absence is not a factual refutation."
        )
    return ProvenanceCase(
        id=case.id, numeric_reference_status=parent.numeric.status,
        original_literal_status=literal, provenance_kind=kind, formula=formula,
        advisory_flags_from_m2p6=[flag.code for flag in parent.flags],
        explanation=explanation,
    )


def audit_derived(
    dataset: Path, run_dir: Path, prior_audit_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Revalidate entire original experiment and parent audit, then write v2."""
    replay_score(dataset, run_dir)
    manifest_file = run_dir / "manifest.json"
    predictions_file = run_dir / "predictions.jsonl"
    evaluation_file = run_dir / "evaluation.json"
    parent_file = prior_audit_dir / "business_audit.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    parent = json.loads(parent_file.read_text(encoding="utf-8"))
    hashes = {
        "source_manifest_sha256": _sha(manifest_file),
        "source_predictions_sha256": _sha(predictions_file),
        "source_evaluation_sha256": _sha(evaluation_file),
    }
    dataset_sha, all_cases = load_data(dataset)
    if parent.get("schema_version") != PARENT_SCHEMA:
        raise ValueError("Expected unchanged M2.6 parent audit schema")
    if parent.get("source_dataset_sha256") != dataset_sha:
        raise ValueError("Parent audit dataset SHA mismatch")
    for key, expected in hashes.items():
        if parent.get(key) != expected:
            raise ValueError(f"Parent audit {key} mismatch")
    previous_saved = json.loads(evaluation_file.read_text(encoding="utf-8"))
    recomputed = replay_score(dataset, run_dir)
    if previous_saved != recomputed:
        raise ValueError("Captured FinanceBench evaluation was modified")
    cases = sample(all_cases, manifest["sample_count"], manifest["seed"])
    records = [
        json.loads(line) for line in predictions_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parents = parent.get("case_results")
    if not isinstance(parents, list) or len(parents) != len(cases):
        raise ValueError("Parent audit case count mismatch")
    outputs = []
    for case, record, stored in zip(cases, records, parents):
        model = FinanceAnswer.model_validate(record["prediction"])
        reread = audit_case(case, model).model_dump(mode="json")
        if reread != stored:
            raise ValueError(f"Parent audit case tampered: {case.id}")
        outputs.append(classify_case(case, model, AuditCase.model_validate(stored)))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "sample_size": len(outputs),
        "dataset_sha256": dataset_sha,
        "source_manifest_sha256": hashes["source_manifest_sha256"],
        "source_predictions_sha256": hashes["source_predictions_sha256"],
        "source_evaluation_sha256": hashes["source_evaluation_sha256"],
        "parent_audit_sha256": _sha(parent_file),
        "provenance_kind_counts": dict(sorted(Counter(x.provenance_kind for x in outputs).items())),
        "formula_relation_counts": dict(sorted(Counter(
            x.formula.arithmetic_relation for x in outputs if x.formula is not None
        ).items())),
        "cases": [x.model_dump(mode="json") for x in outputs],
        "overall_business_accuracy": None,
        "publication_ready": False,
        "independently_human_validated": False,
        "warning": (
            "Arithmetic and numeric occurrence are diagnostic candidates only. "
            "This is not passage entailment, metric-definition validation, "
            "human annotation, or an accuracy improvement evaluation."
        ),
    }
    # Never write anything until original and parent were fully validated.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "arithmetic_provenance.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="M2.6.1 offline, read-only bounded arithmetic audit"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--prior-audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit_derived(
        args.dataset, args.run_dir, args.prior_audit_dir, args.output_dir
    )
    print(json.dumps({
        "schema_version": result["schema_version"],
        "sample_size": result["sample_size"],
        "provenance_kind_counts": result["provenance_kind_counts"],
        "formula_relation_counts": result["formula_relation_counts"],
        "overall_business_accuracy": None,
        "publication_ready": False,
        "sidecar_path": str(args.output_dir / "arithmetic_provenance.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
