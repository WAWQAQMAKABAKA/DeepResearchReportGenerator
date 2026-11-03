"""M1.1 CLI and optional real LangGraph integration checks."""
import json
import subprocess
import sys

import pytest

from deepresearch_v2.cli import build_fixture_report
from deepresearch_v2.core import ResearchReport


def test_headless_cli_direct_files_and_determinism(tmp_path):
    target = tmp_path / "fixture"
    command = [sys.executable, "-m", "deepresearch_v2", "--engine", "direct", "--output-dir", str(target)]
    first = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "accepted claims: 5" in first.stdout
    json_path, md_path = target / "report.json", target / "report.md"
    raw_json, raw_md = json_path.read_bytes(), md_path.read_bytes()
    validated = ResearchReport.model_validate(json.loads(raw_json))
    assert len(validated.profiles) == 3
    assert len(validated.claims) == 5
    assert len(validated.sources) == 3
    assert b"SYNTHETIC OFFLINE FIXTURE" in raw_md
    assert b"Not established" in raw_md
    assert b"[S1](https://fixture.example/1)" in raw_md
    subprocess.run(command, capture_output=True, text=True, check=True)
    assert raw_json == json_path.read_bytes()
    assert raw_md == md_path.read_bytes()


def test_core_fixture_output_deterministic():
    first = build_fixture_report("direct").model_dump_json()
    assert first == build_fixture_report("direct").model_dump_json()


def test_graph_matches_direct_pipeline_when_installed():
    pytest.importorskip("langgraph")
    direct = build_fixture_report("direct")
    graphed = build_fixture_report("graph")
    assert graphed.model_dump(mode="json") == direct.model_dump(mode="json")


def test_graph_cli_files_when_installed(tmp_path):
    pytest.importorskip("langgraph")
    output = tmp_path / "graph"
    result = subprocess.run(
        [sys.executable, "-m", "deepresearch_v2", "--engine", "graph", "--output-dir", str(output)],
        capture_output=True, text=True, check=True
    )
    assert "Engine: graph" in result.stdout
    assert ResearchReport.model_validate_json((output / "report.json").read_text()).profiles[0].company == "AlphaCode"
