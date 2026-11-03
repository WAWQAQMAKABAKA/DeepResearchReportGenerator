"""Reproducible headless M1.1 entry point; synthetic fixture only."""
from __future__ import annotations

import argparse
from pathlib import Path

from .core import ResearchRequest, markdown_table, run_research
from .example import COMPANIES, QUESTION, FixtureExtractor, FixtureRetriever


def build_fixture_report(engine: str):
    """Run the same fixtures through direct Python or the actual LangGraph graph."""
    request = ResearchRequest(question=QUESTION, companies=COMPANIES)
    retriever, extractor = FixtureRetriever(), FixtureExtractor()
    if engine == "graph":
        from .graph import build_graph
        return build_graph(retriever, extractor).invoke({"request": request})["report"]
    if engine == "direct":
        return run_research(request, retriever, extractor)
    raise ValueError(f"Unknown engine: {engine}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic M1.1 evidence pipeline, no UI or API keys")
    parser.add_argument("--engine", choices=["graph", "direct"], default="graph",
                        help="Use real LangGraph by default; direct mode isolates the core")
    parser.add_argument("--output-dir", type=Path, default=Path("build/m1-fixture"),
                        help="Directory for report.json and report.md")
    args = parser.parse_args(argv)

    report = build_fixture_report(args.engine)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "*SYNTHETIC OFFLINE FIXTURE — not real company intelligence.*\n\n"
        + markdown_table(report) + "\n", encoding="utf-8"
    )
    print(f"Engine: {args.engine}; companies: {len(report.profiles)}; "
          f"accepted claims: {len(report.claims)}; rejected candidates: {report.rejected_candidates}")
    print(f"Written: {output_dir / 'report.json'}")
    print(f"Written: {output_dir / 'report.md'}")
    return 0
