"""M2.6 offline business-answer diagnostics over a captured FinanceBench run.

This layer NEVER labels free-text answers as factually correct or false.
It emits transparent, bounded scalar and evidence-literal diagnostics and
conservative interpretation warnings. It does not issue model calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .financebench_benchmark import (
    Case, Excerpt, FinanceAnswer, load_data, replay_score, sample,
)

SCHEMA_VERSION = "m2.6-business-audit-v1"
NumericStatus = Literal[
    "exact", "within_reference_precision", "different", "not_comparable"
]
ValueSource = Literal["explicit", "question_context", "untyped"]


class AuditFlag(BaseModel):
    code: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class NumericCheck(BaseModel):
    status: NumericStatus
    predicted: str | None
    reference: str | None
    canonical_unit: str | None
    absolute_difference: str | None = None
    reference_half_step: str | None = None
    note: str = Field(min_length=1)


class AuditCase(BaseModel):
    id: str
    company: str
    question_type: str
    question: str
    reference_answer: str
    model_answer: str
    cited_excerpts: list[str]
    cited_document_pages: list[str]
    numeric: NumericCheck
    answer_numeric_literal_in_cited_evidence: Literal[
        "found", "not_found", "not_applicable"
    ]
    flags: list[AuditFlag]
    semantic_correctness: Literal["unverified"] = "unverified"
    publication_ready: Literal[False] = False


@dataclass(frozen=True)
class Measurement:
    # Comparable values in base units: USD, percent, percentage points, count,
    # or USD per share. Never compare different kinds.
    amount: Decimal
    unit: str
    resolution: Decimal
    source: ValueSource


NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
FULL_NUMBER = re.compile(rf"^\s*(?P<currency>\$)?(?P<minus>-)?(?P<num>{NUM})\s*$")
QUANTITY = re.compile(
    rf"(?<![\w])(?P<currency>\$)?(?P<minus>-)?(?P<num>{NUM})"
    r"\s*(?P<suffix>percentage\s+points?|pp\b|%|"
    r"(?:USD\s+)?(?:billions?|millions?|thousands?)\b|"
    r"times\b|per\s+share\b)",
    flags=re.IGNORECASE,
)
SOURCE_NUMBER = re.compile(rf"(?<![\w])(?<!\d)(?P<num>-?{NUM})(?![\w])")


def _text_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decimal_num(text: str, negative: bool) -> tuple[Decimal, Decimal]:
    cleaned = text.replace(",", "")
    numeric = Decimal(cleaned)
    if negative:
        numeric = -numeric
    resolution = Decimal(1).scaleb(-len(cleaned.partition(".")[2]))
    return numeric, resolution


def _context_from_question(question: str) -> tuple[str, Decimal] | None:
    """Only infer scale from unambiguous explicit instructions."""
    q = question.casefold()
    if re.search(r"(?:in|answer in|units of)\s+usd\s+billions?\b", q):
        return "usd", Decimal(10) ** 9
    if re.search(r"(?:in|answer in|units of)\s+usd\s+millions?\b", q):
        return "usd", Decimal(10) ** 6
    if re.search(r"(?:in|answer in|units of)\s+usd\s+thousands?\b", q):
        return "usd", Decimal(10) ** 3
    return None


def _from_parts(currency: str | None, number: str, neg: str | None,
                suffix: str, question: str) -> Measurement | None:
    value, resolution = _decimal_num(number, bool(neg))
    unit_suffix = suffix.casefold().strip()
    kind: str
    scale = Decimal(1)
    source: ValueSource = "explicit"
    if unit_suffix in {"%", "percentage point", "percentage points", "pp"}:
        kind = "percent" if unit_suffix == "%" else "percentage_points"
    elif unit_suffix == "times":
        kind = "times"
    elif unit_suffix == "per share":
        if not currency:
            return None  # per-share count versus per-share dollars is ambiguous
        kind = "usd_per_share"
    elif re.search(r"(?:billions?|millions?|thousands?)$", unit_suffix):
        if not currency and not unit_suffix.startswith("usd"):
            # E.g. "2 million users" is not necessarily USD.
            return None
        kind = "usd"
        if "billion" in unit_suffix:
            scale = Decimal(10) ** 9
        elif "million" in unit_suffix:
            scale = Decimal(10) ** 6
        else:
            scale = Decimal(10) ** 3
    else:
        kind = "usd" if currency else "unitless"
        if kind == "usd" or _context_from_question(question):
            context = _context_from_question(question)
            if context:
                kind, scale = context
                source = "question_context"
        else:
            source = "untyped"
    return Measurement(
        amount=value * scale, resolution=resolution * scale,
        unit=kind, source=source,
    )


def parse_measurement(answer: str, question: str) -> Measurement | None:
    """Permit exactly ONE explicit measurement, or a fully standalone number.

    Prevent picking an arbitrary number from multi-number narrative answers.
    Fiscal-year digits do not match a unit-bearing quantity.
    """
    matches = list(QUANTITY.finditer(answer))
    if len(matches) == 1:
        m = matches[0]
        return _from_parts(
            m.group("currency"), m.group("num"), m.group("minus"),
            m.group("suffix"), question,
        )
    if len(matches) > 1:
        return None
    whole = FULL_NUMBER.fullmatch(answer)
    if whole is None:
        return None
    return _from_parts(
        whole.group("currency"), whole.group("num"),
        whole.group("minus"), "", question,
    )


def compare_numeric(reference: str, prediction: str, question: str) -> NumericCheck:
    gold = parse_measurement(reference, question)
    pred = parse_measurement(prediction, question)
    if gold is None or pred is None:
        return NumericCheck(
            status="not_comparable", predicted=str(pred.amount) if pred else None,
            reference=str(gold.amount) if gold else None, canonical_unit=None,
            note="More than one value, untyped value, or prose: no automatic correctness grade.",
        )
    if gold.unit != pred.unit:
        return NumericCheck(
            status="not_comparable",
            predicted=str(pred.amount), reference=str(gold.amount),
            canonical_unit=None,
            note="Units differ; no implicit unit conversion is assumed.",
        )
    diff = abs(pred.amount - gold.amount)
    half = gold.resolution / Decimal(2)
    if diff == 0:
        status: NumericStatus = "exact"
    elif diff <= half:
        status = "within_reference_precision"
    else:
        status = "different"
    return NumericCheck(
        status=status,
        predicted=str(pred.amount), reference=str(gold.amount),
        canonical_unit=gold.unit, absolute_difference=str(diff),
        reference_half_step=str(half),
        note=("Numeric relation to the published reference only, not proof of "
              "financial correctness or correct metric/period selection."),
    )


def source_literal_check(
    prediction: FinanceAnswer, case: Case, measurement: Measurement | None
) -> Literal["found", "not_found", "not_applicable"]:
    if not prediction.evidence_ids or measurement is None:
        return "not_applicable"
    # Literal lookup, not mathematical derivation or semantic evidence support.
    numbers: set[Decimal] = set()
    for excerpt in case.evidence:
        if excerpt.id in prediction.evidence_ids:
            for match in SOURCE_NUMBER.finditer(excerpt.text):
                numbers.add(Decimal(match.group("num").replace(",", "")))
    # Many FinanceBench tables are in millions: a $8.738 billion answer can
    # be literally backed by "8,738" on a table labeled (in millions).
    # Only count this simple scale conversion when the original question
    # explicitly identifies USD millions/billions, or excerpt has (in millions).
    probes = {measurement.amount}
    if measurement.unit == "usd":
        if _context_from_question(case.question) or any(
            "in millions" in e.text.casefold() or "dollars in millions" in e.text.casefold()
            for e in case.evidence if e.id in prediction.evidence_ids
        ):
            probes.add(measurement.amount / (Decimal(10) ** 6))
        if _context_from_question(case.question):
            probes.add(measurement.amount / (Decimal(10) ** 9))
    if any(x in numbers for x in probes):
        return "found"
    return "not_found"


def _flag(code: str, message: str) -> AuditFlag:
    return AuditFlag(code=code, explanation=message)


def flags_for(case: Case, prediction: FinanceAnswer, numeric: NumericCheck,
              literal_status: str) -> list[AuditFlag]:
    q = case.question.casefold()
    r = prediction.rationale.casefold()
    cited = [ex for ex in case.evidence if ex.id in prediction.evidence_ids]
    flags: list[AuditFlag] = []
    if numeric.status == "different":
        flags.append(_flag(
            "numeric_reference_disagreement",
            "Model and reference scalar differ beyond half of the reference's last displayed digit; inspect source and rounding. Do not assume model is wrong.",
        ))
    if numeric.status == "not_comparable":
        flags.append(_flag(
            "numeric_ungraded",
            "An automatic numerical equivalence judgment is not available for this answer.",
        ))
    if literal_status == "not_found":
        flags.append(_flag(
            "answer_value_not_literal_in_cited_excerpt",
            "The answer's numeric value was not directly located in cited text; it may be derived through calculation or require review.",
        ))
    if re.search(r"operating\s+margin", q) and (
        ("pretax" in r or "pre-tax" in r) and (
            "proxy" in r or "closest" in r or "used as" in r
        )
    ):
        flags.append(_flag(
            "metric_definition_proxy",
            "Pretax income used as an operating-income/margin substitute; validate metric suitability, especially for financial services.",
        ))
    if ("liquidat" in q or "bankrupt" in q) and (
        "tangible book value" in r or "tbvps" in r or "book value" in r
    ):
        flags.append(_flag(
            "liquidation_book_value_assumption",
            "Book value per share is not established liquidation proceeds; this comparison requires an explicit financial caveat.",
        ))
    if "income statements" in q and cited and all(
        ("restructuring" in ex.text.casefold() and
         "note" in ex.text[:120].casefold() and
         "consolidated statements of income" not in ex.text.casefold())
        for ex in cited
    ):
        flags.append(_flag(
            "requested_statement_vs_note",
            "Evidence comes from a financial statement note, while the question requests an amount directly outlined in the income statement.",
        ))
    return flags


def audit_case(case: Case, prediction: FinanceAnswer) -> AuditCase:
    ids = {e.id for e in case.evidence}
    if len(set(prediction.evidence_ids)) != len(prediction.evidence_ids):
        raise ValueError(f"Duplicate evidence IDs: {case.id}")
    if not set(prediction.evidence_ids) <= ids:
        raise ValueError(f"Unknown evidence IDs: {case.id}")
    numeric = compare_numeric(case.gold_answer, prediction.answer, case.question)
    model_measurement = parse_measurement(prediction.answer, case.question)
    literal = source_literal_check(prediction, case, model_measurement)
    sources = [
        f"{e.doc_name}:p{e.page}" for e in case.evidence
        if e.id in prediction.evidence_ids
    ]
    return AuditCase(
        id=case.id, company=case.company, question_type=case.question_type,
        question=case.question,
        reference_answer=case.gold_answer, model_answer=prediction.answer,
        cited_excerpts=prediction.evidence_ids,
        cited_document_pages=sources,
        numeric=numeric,
        answer_numeric_literal_in_cited_evidence=literal,
        flags=flags_for(case, prediction, numeric, literal),
    )


def audit_existing_run(
    dataset: Path, run_dir: Path, out_dir: Path
) -> dict[str, Any]:
    """Fail closed on changed data or run, then write a NEW immutable sidecar."""
    recomputed = replay_score(dataset, run_dir)
    saved = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    if saved != recomputed:
        raise ValueError("Captured evaluation.json does not match recorded predictions")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    _, rows = load_data(dataset)
    selected = sample(rows, manifest["sample_count"], manifest["seed"])
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    # replay_score already validated data SHA, sample selection and IDs.
    audited = [
        audit_case(case, FinanceAnswer.model_validate(record["prediction"]))
        for case, record in zip(selected, records)
    ]
    counts = Counter(c.numeric.status for c in audited)
    flags = Counter(f.code for c in audited for f in c.flags)
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset_sha256": manifest["dataset_sha256"],
        "source_manifest_sha256": _text_sha(run_dir / "manifest.json"),
        "source_predictions_sha256": _text_sha(run_dir / "predictions.jsonl"),
        "source_evaluation_sha256": _text_sha(run_dir / "evaluation.json"),
        "sample_size": len(audited),
        "numeric_status_counts": dict(sorted(counts.items())),
        "advisory_flag_counts": dict(sorted(flags.items())),
        "case_results": [c.model_dump(mode="json") for c in audited],
        "independently_human_validated": False,
        "overall_business_accuracy": None,
        "publication_ready": False,
        "warning": (
            "Numeric similarity and source literal occurrence are not evidence entailment. "
            "Financial interpretations and ambiguous benchmark references remain unverified."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json = json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (out_dir / "business_audit.json").write_text(_write_json, encoding="utf-8")
    return sidecar


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="M2.6 deterministic evidence/units/reference audit of completed FinanceBench run"
    )
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args(argv)
    result = audit_existing_run(args.dataset, args.run_dir, args.output_dir)
    summary = {
        "schema_version": result["schema_version"],
        "sample_size": result["sample_size"],
        "numeric_status_counts": result["numeric_status_counts"],
        "advisory_flag_counts": result["advisory_flag_counts"],
        "overall_business_accuracy": None,
        "publication_ready": False,
        "sidecar_path": str(args.output_dir / "business_audit.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
