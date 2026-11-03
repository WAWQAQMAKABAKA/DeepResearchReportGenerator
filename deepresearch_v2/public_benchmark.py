"""Replay a small, real-world public-evidence snapshot through the M1 graph.

Sources: short verified excerpts from public official pages. Extraction is human
curated and deliberately NOT presented as a successful autonomous model run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from .core import Candidate, ExtractionBatch, ResearchReport, ResearchRequest, Source, markdown_table, run_research

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "ai_coding_agents_2026-11-03.json"


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["capture_method"] == "manual_verified_web_excerpts"
    assert len(data["sources"]) == len({s["id"] for s in data["sources"]})
    return data


class SnapshotRetriever:
    def __init__(self, snapshot: dict):
        self.snapshot = snapshot

    def retrieve(self, request: ResearchRequest) -> list[Source]:
        if request != ResearchRequest.model_validate(self.snapshot["request"]):
            raise ValueError("Snapshot request does not match the pinned study")
        retrieved_at = datetime.fromisoformat(self.snapshot["captured_at"].replace("Z", "+00:00"))
        return [Source(id=item["id"], company=item["company"], url=item["url"],
                       title=item["title"], text=item["text"],
                       retrieved_at=retrieved_at, provider="public_web_excerpt")
                for item in self.snapshot["sources"]]


class CuratedExtractor:
    def __init__(self, snapshot: dict):
        self.snapshot = snapshot

    def extract(self, request: ResearchRequest, sources: list[Source]) -> ExtractionBatch:
        # A deterministic HUMAN-REVIEWED oracle, not an automated extractor.
        return ExtractionBatch(items=[Candidate.model_validate(item) for item in self.snapshot["candidates"]])


def run_public_snapshot(engine: str = "graph", path: Path = SNAPSHOT_PATH) -> ResearchReport:
    snapshot = load_snapshot(path)
    request = ResearchRequest.model_validate(snapshot["request"])
    retriever, extractor = SnapshotRetriever(snapshot), CuratedExtractor(snapshot)
    if engine == "direct":
        return run_research(request, retriever, extractor)
    if engine == "graph":
        from .graph import build_graph
        return build_graph(retriever, extractor).invoke({"request": request})["report"]
    raise ValueError("Unknown engine")


def evaluate(report: ResearchReport, snapshot: dict) -> dict:
    """Structural coverage metrics, NOT semantic support or model accuracy."""
    total_fields = 8 * len(report.profiles)
    covered = sum(len(p.claims_by_field) for p in report.profiles)
    unknown = total_fields - covered
    return {
        "study": snapshot["snapshot_id"],
        "provenance": snapshot["capture_method"],
        "extraction": "human_curated_not_llm",
        "company_count": len(report.profiles),
        "source_count": len(report.sources),
        "accepted_claims": len(report.claims),
        "rejected_candidates": report.rejected_candidates,
        "populated_company_fields": covered,
        "company_fields_total": total_fields,
        "unpopulated_company_fields": unknown,
        "semantic_support_assessed": False,
        "live_model_assessed": False,
        "source_digest_sha256": {s.id: hashlib.sha256(s.text.encode("utf-8")).hexdigest() for s in report.sources},
        "historical_price_warning": "2024 Devin $500/month mention deliberately not used as November 2026 current price",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay verified public excerpts; no network or model key required")
    parser.add_argument("--engine", choices=["graph", "direct"], default="graph")
    parser.add_argument("--output-dir", type=Path, default=Path("build/m2-public-snapshot"))
    args = parser.parse_args(argv)
    snap = load_snapshot()
    report = run_public_snapshot(args.engine)
    results = evaluate(report, snap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(
        "*Source excerpt replay; extraction manually curated, not LLM-generated. Pricing as observed on 2026-11-03.*\n\n"
        + markdown_table(report) + "\n", encoding="utf-8")
    (args.output_dir / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"{args.engine}: {results['source_count']} sources, {results['accepted_claims']} claims, "
          f"{results['populated_company_fields']}/{results['company_fields_total']} fields; "
          f"{results['rejected_candidates']} rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
