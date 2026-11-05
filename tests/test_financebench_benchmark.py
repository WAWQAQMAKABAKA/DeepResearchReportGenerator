"""FinanceBench financial QA regression tests; every test row is SYNTHETIC."""
from __future__ import annotations
import json
from decimal import Decimal

import pytest
from pydantic import ValidationError
from deepresearch_v2.financebench_benchmark import (
    TYPES, VERSION, FinanceAnswer, blind, load_data, main,
    parse_scalar, prompt, replay_score, run, sample, score,
)


def make_data(tmp_path):
    rows = []
    for i in range(12):
        kind = TYPES[i % 3]
        rows.append({
            "financebench_id": f"financebench_id_{i:05d}",
            "company": "Synthetic Corp",
            "doc_name": "SYNTH_2024_10K",
            "question_type": kind,
            "question": f"What is Synthetic Corp metric {i}?",
            "answer": "$1000.00" if kind == "metrics-generated" else f"Synthetic explanation {i}",
            "justification": f"GOLD JUSTIFICATION {i} MUST NEVER LEAK",
            "evidence": [{
                "evidence_text": f"Synthetic line item {i}: $1,000.",
                "doc_name": "SYNTH_2024_10K",
                "evidence_page_num": i,
                "evidence_text_full_page": "SECRET FULL PAGE NOT USED",
            }],
        })
    out = tmp_path / "financebench_open_source.jsonl"
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return out


class DummyPredictor:
    def __init__(self):
        self.sent_prompts = []

    def predict(self, question):
        self.sent_prompts.append(prompt(question))
        return (FinanceAnswer(
            answer="$1,000", rationale="synthetic arithmetic",
            evidence_ids=["E1"]
        ), {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8})


def test_load_actual_document_field_and_balance(tmp_path):
    dataset = make_data(tmp_path)
    digest, rows = load_data(dataset)
    assert len(digest) == 64 and len(rows) == 12
    assert rows[0].evidence[0].doc_name == "SYNTH_2024_10K"
    a, b = sample(rows, 6), sample(rows, 6)
    assert [c.id for c in a] == [c.id for c in b]
    assert {t: sum(c.question_type == t for c in a) for t in TYPES} == {t: 2 for t in TYPES}
    with pytest.raises(ValueError, match="count"):
        sample(rows, 31)
    with pytest.raises(ValueError, match="count"):
        sample(rows, 13)
    row = json.loads(dataset.read_text().splitlines()[0])
    row["evidence"][0]["evidence_page_num"] = -2
    dataset.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValidationError):
        load_data(dataset)


def test_prompt_withholds_gold_and_full_page(tmp_path):
    _, rows = load_data(make_data(tmp_path))
    rendered = json.dumps(prompt(blind(rows[0])))
    assert "Synthetic line item 0" in rendered
    assert "GOLD JUSTIFICATION" not in rendered
    assert "SECRET FULL PAGE" not in rendered
    assert '"gold_answer"' not in rendered
    with pytest.raises(ValueError, match="over budget"):
        blind(rows[0], char_limit=1)


def test_strict_scalar_and_unscored_prose(tmp_path):
    assert parse_scalar("$1,000.00") == ("usd", "$", Decimal("1000.00"))
    assert parse_scalar("$1000") == ("usd", "$", Decimal("1000"))
    assert parse_scalar("10%") == ("percent", "", Decimal("10"))
    assert parse_scalar("1000") == ("plain", "", Decimal("1000"))
    assert parse_scalar("About $1000") is None
    _, data = load_data(make_data(tmp_path))
    chosen = sample(data, 6)
    records = []
    for case in chosen:
        output = "$1,000" if case.question_type == "metrics-generated" else "Different words"
        records.append({"id": case.id, "prediction": {
            "answer": output, "rationale": "synthetic", "evidence_ids": ["E1"]
        }})
    metrics = score(chosen, records)
    assert metrics["sample_size"] == 6
    assert metrics["strict_scalar_diagnostic"]["gold_scalar_cases"] == 2
    assert metrics["strict_scalar_diagnostic"]["comparable_cases"] == 2
    assert metrics["strict_scalar_diagnostic"]["equal"] == 2
    assert metrics["strict_scalar_diagnostic"]["different"] == 0
    assert metrics["requires_semantic_review_count"] == 4
    assert metrics["official_financebench_score"] is None
    with pytest.raises(ValueError, match="match selected"):
        score(chosen, list(reversed(records)))
    with pytest.raises(ValidationError):
        FinanceAnswer(answer="", rationale="synthetic")


def test_missing_currency_is_not_labeled_wrong(tmp_path):
    _, data = load_data(make_data(tmp_path))
    chosen = sample(data, 3)
    records = []
    for case in chosen:
        output = "1000" if case.question_type == "metrics-generated" else case.gold_answer
        records.append({"id": case.id, "prediction": {
            "answer": output, "rationale": "synthetic", "evidence_ids": []
        }})
    result = score(chosen, records)
    assert result["strict_scalar_diagnostic"]["gold_scalar_cases"] == 1
    assert result["strict_scalar_diagnostic"]["comparable_cases"] == 0
    assert result["requires_semantic_review_count"] == 1
    assert result["normalized_exact_string_agreement"]["count"] == 2


def test_mock_model_run_immutable_and_replays(tmp_path):
    path = make_data(tmp_path)
    output = tmp_path / "results"
    model = DummyPredictor()
    first = run(path, output, model, count=6, model="mock")
    assert first["sample_size"] == 6
    assert first["tokens"]["total_tokens"] == 48
    assert replay_score(path, output) == first
    assert len(model.sent_prompts) == 6
    for request in model.sent_prompts:
        assert "GOLD JUSTIFICATION" not in json.dumps(request)
        assert '"gold_answer"' not in json.dumps(request)
    with pytest.raises(FileExistsError):
        run(path, output, model, count=6)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        replay_score(path, output)


def test_failed_model_run_remains_unscored(tmp_path):
    dataset = make_data(tmp_path)
    class FailModel:
        counter = 0
        def predict(self, q):
            self.counter += 1
            if self.counter == 3:
                raise RuntimeError("Synthetic outage")
            return FinanceAnswer(answer="No", rationale="synthetic"), {}
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Synthetic outage"):
        run(dataset, out, FailModel(), count=6)
    status = json.loads((out / "status.json").read_text())
    assert status["completed"] == 2
    assert status["status"] == "failed"
    assert not (out / "evaluation.json").exists()
    with pytest.raises(ValueError, match="Incomplete"):
        replay_score(dataset, out)


def test_no_api_inspection_cli(tmp_path, capsys):
    assert main(["inspect", "--dataset", str(make_data(tmp_path)), "--count", "6"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model_calls"] == 0
    assert data["dataset_rows"] == 12
    assert data["selected"] == 6
    assert data["gold_scalar_count"] == 2
    assert VERSION.startswith("m2.5b")
