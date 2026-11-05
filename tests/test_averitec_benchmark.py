"""M2.5a offline tests: synthetic AVeriTeC-shaped data, NO benchmark labels fabricated as real."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from deepresearch_v2.averitec_benchmark import (
    LABELS, Prediction, UPSTREAM_REVISION, VERSION,
    blind, load_dev, main, prompt, replay_score, run, sample, score,
)


def fixture(tmp_path):
    rows = []
    for i in range(20):
        rows.append({
            "claim": f"Synthetic claim {i}",
            "label": LABELS[i % 4],
            "claim_date": "2024-01-01",
            "speaker": "Test speaker",
            "justification": f"GOLD JUSTIFICATION LEAK {i}",
            "fact_checking_article": "https://example.test/gold_label_in_url",
            "questions": [{
                "question": f"Evidence question {i}",
                "answers": [{
                    "answer": f"Synthetic answer {i}",
                    "source_url": f"https://example.test/source_{i}",
                    "answer_type": "extractive",
                }],
            }],
        })
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


class FakePredictor:
    def __init__(self):
        self.prompts = []

    def predict(self, item):
        self.prompts.append(prompt(item))
        return Prediction(
            verdict="Supported", explanation="Synthetic prediction",
            evidence_ids=[item.evidence[0]["id"]],
        ), {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16}


def test_schema_stratification_and_refusal(tmp_path):
    source = fixture(tmp_path)
    digest, examples = load_dev(source)
    assert len(digest) == 64
    one = sample(examples, 12, 5243)
    two = sample(examples, 12, 5243)
    assert [e.example_id for e in one] == [e.example_id for e in two]
    assert {label: sum(e.label == label for e in one) for label in LABELS} == {
        label: 3 for label in LABELS
    }
    with pytest.raises(ValueError, match="between"):
        sample(examples, 51)
    with pytest.raises(ValueError, match="within"):
        sample(examples, 21)
    obj = json.loads(source.read_text())
    obj[0]["label"] = "FAKE_LABEL"
    source.write_text(json.dumps(obj))
    with pytest.raises(ValidationError):
        load_dev(source)


def test_gold_label_and_fact_check_article_never_sent_to_model(tmp_path):
    _, examples = load_dev(fixture(tmp_path))
    item = blind(examples[0])
    rendered = json.dumps(prompt(item), ensure_ascii=False)
    assert "Synthetic claim 0" in rendered
    assert "Evidence question 0" in rendered
    assert "GOLD JUSTIFICATION LEAK" not in rendered
    assert "gold_label_in_url" not in rendered
    assert "source_0" not in rendered
    assert '"label"' not in rendered
    assert '"fact_checking_article"' not in rendered


def test_four_class_deterministic_metrics(tmp_path):
    _, examples = load_dev(fixture(tmp_path))
    chosen = sample(examples, 8)
    records = [{
        "id": ex.example_id, "prediction": {
            "verdict": "Supported", "explanation": "Synthetic"
        }
    } for ex in chosen]
    metrics = score(chosen, records)
    assert metrics["accuracy"] == 0.25
    assert metrics["macro_f1"] == pytest.approx(0.1)
    assert metrics["always_supported_baseline"]["accuracy"] == 0.25
    assert metrics["official_averitec_score"] is None
    assert metrics["confusion_matrix"]["Supported"]["Supported"] == 2
    assert len(metrics["errors"]) == 6
    with pytest.raises(ValueError, match="exact sample"):
        score(chosen, list(reversed(records)))
    with pytest.raises(ValidationError):
        Prediction(verdict="Invalid", explanation="Synthetic")


def test_model_stub_run_is_reproducible_and_input_digest_bound(tmp_path):
    src = fixture(tmp_path)
    folder = tmp_path / "run"
    predictor = FakePredictor()
    metrics = run(src, folder, predictor, count=12, seed=50, model="mock")
    assert metrics["sample_size"] == 12
    assert metrics["accuracy"] == 0.25
    assert metrics["tokens"]["total_tokens"] == 192
    assert len(predictor.prompts) == 12
    assert len((folder / "predictions.jsonl").read_text().splitlines()) == 12
    assert replay_score(src, folder) == metrics
    with pytest.raises(FileExistsError):
        run(src, folder, predictor, count=12, seed=50)
    src.write_bytes(src.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        replay_score(src, folder)


def test_failure_is_not_counted_as_completed_benchmark(tmp_path):
    src = fixture(tmp_path)

    class Fails:
        calls = 0

        def predict(self, example):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("Synthetic service error")
            return Prediction(verdict="Supported", explanation="Synthetic"), {}

    folder = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="Synthetic"):
        run(src, folder, Fails(), count=8)
    status = json.loads((folder / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["completed"] == 2
    assert not (folder / "evaluation.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        replay_score(src, folder)


def test_offline_inspect_cli_without_keys(tmp_path, capsys):
    src = fixture(tmp_path)
    assert main(["inspect", "--dataset", str(src), "--count", "8"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["selected"] == 8
    assert out["model_calls"] == 0
    assert out["dataset_rows"] == 20
    assert VERSION.startswith("m2.5a")
    assert len(UPSTREAM_REVISION) == 40
