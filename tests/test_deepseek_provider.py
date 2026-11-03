"""Offline DeepSeek provider qualification: no real API calls."""
from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

from deepresearch_v2.core import ExtractionBatch, ResearchRequest
from deepresearch_v2.live import LangChainExtractor
from deepresearch_v2.public_benchmark import SnapshotRetriever, load_snapshot
from deepresearch_v2.qualification import _parser, _read_json, qualify_run, replay_run


class FakeRun:
    def invoke(self, messages):
        query = messages[1]["content"]
        items = ([{"company": "Cursor", "field": "product",
                   "value": "Cursor's AI assistant", "source_id": "S1",
                   "quote": "Agent is Cursor's AI assistant."}]
                 if "COMPANY: Cursor\n" in query else [])
        return {"parsed": {"items": items},
                "raw": SimpleNamespace(content="test response",
                    usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
                "parsing_error": None}


class FakeDeepSeek:
    instances = []
    structured_calls = []

    def __init__(self, *, model, temperature):
        self.instances.append((model, temperature))

    def with_structured_output(self, schema, *, method, include_raw):
        self.structured_calls.append((schema, method, include_raw))
        return FakeRun()


def _inject(monkeypatch):
    FakeDeepSeek.instances.clear()
    FakeDeepSeek.structured_calls.clear()
    monkeypatch.setitem(sys.modules, "langchain_deepseek",
                        SimpleNamespace(ChatDeepSeek=FakeDeepSeek))


def test_cli_accepts_deepseek():
    args = _parser().parse_args(["extract-fixture", "--model-provider", "deepseek",
                                "--model", "deepseek-flash", "--output-dir", "build/test"])
    assert args.model_provider == "deepseek"
    assert args.model == "deepseek-flash"


def test_structured_deepseek_binding(monkeypatch):
    _inject(monkeypatch)
    adapter = LangChainExtractor(provider="deepseek", model="deepseek-flash")
    assert isinstance(adapter.structured_model, FakeRun)
    assert FakeDeepSeek.instances == [("deepseek-flash", 0)]
    assert FakeDeepSeek.structured_calls == [(ExtractionBatch, "function_calling", True)]


def test_mock_deepseek_capture_trace_and_replay(monkeypatch, tmp_path):
    _inject(monkeypatch)
    snapshot = load_snapshot()
    request = ResearchRequest.model_validate(snapshot["request"])
    adapter = LangChainExtractor(provider="deepseek", model="deepseek-flash")
    source = SnapshotRetriever(snapshot)
    output = tmp_path / "first"
    result = qualify_run(request, source, adapter, output, engine="direct",
                         mode="llm-on-curated-excerpts", gold=snapshot)
    assert result["qualification_status"] == "UNREVIEWED_CLAIMS"
    assert result["semantic_support_evaluated"] is False
    assert result["accepted_claim_count"] == 1
    captured = _read_json(output / "candidates.json")
    assert captured["model"]["provider"] == "deepseek"
    assert captured["model"]["name"] == "deepseek-flash"
    assert len(captured["invocations"]) == 3
    assert captured["invocations"][0]["token_usage"] == {
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15
    }
    assert captured["invocations"][0]["raw_response_sha256"] == hashlib.sha256(
        b"test response"
    ).hexdigest()
    replay = tmp_path / "replay"
    replay_run(output, replay, engine="direct")
    assert (output / "report.json").read_bytes() == (replay / "report.json").read_bytes()
    assert (output / "report.md").read_bytes() == (replay / "report.md").read_bytes()
