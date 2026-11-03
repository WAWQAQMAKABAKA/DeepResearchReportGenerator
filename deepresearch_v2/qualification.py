"""M2.1 headless qualification: capture -> replay -> review.

This harness deliberately distinguishes three different achievements:
(1) an adapter runs, (2) exact candidate/excerpt links are valid, and
(3) a human has judged whether the cited passage actually supports a claim.
Only (1) and (2) are automated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .core import ExtractionBatch, ResearchReport, ResearchRequest, Source, markdown_table, run_research
from .live import LangChainExtractor, PROMPT_VERSION, TavilyRetriever
from .public_benchmark import CuratedExtractor, SnapshotRetriever, load_snapshot

SCHEMA_VERSION = "m2.1-v1"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_file(path: Path, payload: dict[str, Any]) -> None:
    """Write fully validated artifacts atomically; no partial JSON files."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


class ReviewLabel(BaseModel):
    claim_id: str = Field(min_length=1)
    verdict: Literal["supported", "contradicted", "insufficient", "stale", "uncertain"]
    note: str = Field(default="", max_length=1000)


class ManualReview(BaseModel):
    reviewer: str = Field(min_length=1)
    labels: list[ReviewLabel]

    @model_validator(mode="after")
    def unique_ids(self) -> "ManualReview":
        ids = [label.claim_id for label in self.labels]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate reviewed claim ID")
        return self


def score_manual_review(report: ResearchReport, review: ManualReview) -> dict[str, Any]:
    known = {claim.id for claim in report.claims}
    unknown = {label.claim_id for label in review.labels} - known
    if unknown:
        raise ValueError(f"Unknown claim IDs in review: {sorted(unknown)}")
    counts = {v: sum(l.verdict == v for l in review.labels)
              for v in ("supported", "contradicted", "insufficient", "stale", "uncertain")}
    n_reviewed = len(review.labels)
    return {
        "review_status": "complete" if n_reviewed == len(report.claims) else "partial",
        "reviewer": review.reviewer,
        "reviewed_claims": n_reviewed,
        "unreviewed_claims": len(report.claims) - n_reviewed,
        "verdict_counts": counts,
        "support_rate_among_reviewed": counts["supported"] / n_reviewed if n_reviewed else None,
        "all_claims_passage_supported_by_review": (len(report.claims) > 0 and
            n_reviewed == len(report.claims) and counts["supported"] == len(report.claims)),
        "note": "Manual passage-entailment labels, not independent confirmation of a vendor assertion",
    }


def structural_metrics(report: ResearchReport, *, gold: dict[str, Any] | None = None) -> dict[str, Any]:
    fields_total = len(report.profiles) * 8
    present_fields = sum(len(p.claims_by_field) for p in report.profiles)
    metrics: dict[str, Any] = {
        "qualification_status": "UNREVIEWED_CLAIMS",
        "company_count": len(report.profiles),
        "source_count": len(report.sources),
        "accepted_claim_count": len(report.claims),
        "rejected_candidate_count": report.rejected_candidates,
        "populated_fields": present_fields,
        "total_fields": fields_total,
        "missing_fields": fields_total - present_fields,
        "semantic_support_evaluated": False,
        "live_model_accuracy_verified": False,
        "model_output_is_untrusted": True,
    }
    if gold is not None:
        # Diagnostic ONLY. Different wording and paraphrase may be correct but
        # miss exact-match scoring; matching gold is not proof of entailment.
        gold_values = {(_normalized(c["company"]), c["field"], _normalized(c["value"]))
                       for c in gold["candidates"]}
        predicted = {(_normalized(c.company), c.field, _normalized(c.value)) for c in report.claims}
        metrics["gold_exact_alignment"] = {
            "gold_claims": len(gold_values),
            "accepted_claims": len(predicted),
            "matching_value_triplets": len(gold_values & predicted),
            "interpretation": "Lexical agreement with human-curated values, not semantic precision/recall",
        }
    return metrics


class _RecordingRetriever:
    def __init__(self, provider: Any, request: ResearchRequest, run_dir: Path, mode: str):
        self.provider, self.request, self.run_dir, self.mode = provider, request, run_dir, mode

    def retrieve(self, request: ResearchRequest) -> list[Source]:
        sources = self.provider.retrieve(request)
        if len(sources) != len({s.id for s in sources}):
            raise ValueError("Retriever returned duplicate source IDs")
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "runtime_versions": {"python": platform.python_version(),
                                 "pydantic": _version("pydantic"),
                                 "langgraph": _version("langgraph"),
                                 "tavily-python": _version("tavily-python"),
                                 "langchain-ollama": _version("langchain-ollama")},
            "mode": self.mode,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "request": request.model_dump(mode="json"),
            "sources": [s.model_dump(mode="json") for s in sources],
            "source_sha256": {s.id: _digest(s.text) for s in sources},
            "query_log": getattr(self.provider, "search_log", []),
            "source_provenance": getattr(self.provider, "source_provenance", {}),
        }
        _json_file(self.run_dir / "sources.json", snapshot)
        if not sources:
            raise RuntimeError("No usable retrieved sources; extraction aborted, sources.json retained")
        return sources


class _RecordingExtractor:
    def __init__(self, provider: Any, run_dir: Path):
        self.provider, self.run_dir = provider, run_dir

    def extract(self, request: ResearchRequest, sources: list[Source]) -> ExtractionBatch:
        batch = self.provider.extract(request, sources)
        batch = ExtractionBatch.model_validate(batch)
        model = getattr(self.provider, "model", "recorded-or-mock")
        provider = getattr(self.provider, "provider", "fixture")
        # Never persist credentials, env vars, or API tokens.
        _json_file(self.run_dir / "candidates.json", {
            "schema_version": SCHEMA_VERSION,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "model": {"provider": provider, "name": model, "prompt_version": PROMPT_VERSION},
            "batch": batch.model_dump(mode="json"),
            "batch_sha256": _digest(batch.model_dump_json()),
            "source_sha256": {s.id: _digest(s.text) for s in sources},
            "invocations": getattr(self.provider, "invocations", []),
        })
        return batch


def _compose_engine(request: ResearchRequest, retriever: Any, extractor: Any, engine: str) -> ResearchReport:
    if engine == "graph":
        from .graph import build_graph
        return build_graph(retriever, extractor).invoke({"request": request})["report"]
    if engine == "direct":
        return run_research(request, retriever, extractor)
    raise ValueError(f"Unknown engine: {engine}")


def _save_result(report: ResearchReport, run_dir: Path, *, gold: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = structural_metrics(report, gold=gold)
    _json_file(run_dir / "report.json", report.model_dump(mode="json"))
    (run_dir / "report.md").write_text(
        "*Qualification output: source-reported claims are UNREVIEWED, not verified facts.*\n\n"
        + markdown_table(report) + "\n", encoding="utf-8",
    )
    _json_file(run_dir / "evaluation.json", metrics)
    return metrics


def qualify_run(
    request: ResearchRequest,
    retriever: Any,
    extractor: Any,
    run_dir: Path,
    *,
    engine: str = "graph",
    mode: str = "live",
    gold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an immutable run directory. Save sources before trying extraction."""
    run_dir.mkdir(parents=True, exist_ok=False)  # no accidental overwrite
    _json_file(run_dir / "status.json", {"status": "running", "mode": mode, "engine": engine})
    try:
        report = _compose_engine(
            request,
            _RecordingRetriever(retriever, request, run_dir, mode),
            _RecordingExtractor(extractor, run_dir),
            engine,
        )
        metrics = _save_result(report, run_dir, gold=gold)
        _json_file(run_dir / "status.json", {"status": "completed_unreviewed", "mode": mode, "engine": engine})
        return metrics
    except Exception as exc:
        # Don't persist exception messages, which might include secrets or raw model text.
        _json_file(run_dir / "status.json", {
            "status": "failed", "mode": mode, "engine": engine,
            "error_type": type(exc).__name__,
            "retrieval_captured": (run_dir / "sources.json").exists(),
            "extraction_captured": (run_dir / "candidates.json").exists(),
        })
        raise


class _RecordedRetriever:
    def __init__(self, request: ResearchRequest, sources: list[Source]):
        self.expected_request, self.sources = request, sources

    def retrieve(self, request: ResearchRequest) -> list[Source]:
        if request != self.expected_request:
            raise ValueError("Replayed request does not match captured request")
        return list(self.sources)


class _RecordedExtractor:
    def __init__(self, batch: ExtractionBatch):
        self.batch = batch

    def extract(self, request: ResearchRequest, sources: list[Source]) -> ExtractionBatch:
        return self.batch


def load_recorded_run(run_dir: Path) -> tuple[ResearchRequest, list[Source], ExtractionBatch]:
    sr = _read_json(run_dir / "sources.json")
    er = _read_json(run_dir / "candidates.json")
    if sr.get("schema_version") != SCHEMA_VERSION or er.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported snapshot version")
    request = ResearchRequest.model_validate(sr["request"])
    sources = [Source.model_validate(source) for source in sr["sources"]]
    if len({s.id for s in sources}) != len(sources):
        raise ValueError("Duplicate source IDs in recorded run")
    expected_hashes = {s.id: _digest(s.text) for s in sources}
    if sr["source_sha256"] != expected_hashes or er["source_sha256"] != expected_hashes:
        raise ValueError("Recorded source text has changed since extraction")
    batch = ExtractionBatch.model_validate(er["batch"])
    if er["batch_sha256"] != _digest(batch.model_dump_json()):
        raise ValueError("Recorded model candidates were modified")
    return request, sources, batch


def replay_run(run_dir: Path, output_dir: Path, *, engine: str = "graph") -> dict[str, Any]:
    request, sources, batch = load_recorded_run(run_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = _compose_engine(request, _RecordedRetriever(request, sources), _RecordedExtractor(batch), engine)
    return _save_result(report, output_dir)


def review_run(report_path: Path, label_path: Path, output_path: Path) -> dict[str, Any]:
    report = ResearchReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    labels = ManualReview.model_validate(_read_json(label_path))
    score = score_manual_review(report, labels)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _json_file(output_path, score)
    return score


class _LazyLangChainExtractor:
    """Delay endpoint/model initialization until after retrieval is captured."""

    def __init__(self, provider: str, model: str):
        self.provider, self.model = provider, model
        self.invocations: list[dict[str, Any]] = []

    def extract(self, request: ResearchRequest, sources: list[Source]) -> ExtractionBatch:
        extractor = LangChainExtractor(provider=self.provider, model=self.model)
        batch = extractor.extract(request, sources)
        self.invocations = extractor.invocations
        return batch


def _make_model(args: argparse.Namespace) -> _LazyLangChainExtractor:
    return _LazyLangChainExtractor(provider=args.model_provider, model=args.model)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M2.1 reproducible live qualification (no Streamlit)")
    sub = parser.add_subparsers(dest="command", required=True)
    for mode in ("live", "extract-fixture", "extract-recorded"):
        p = sub.add_parser(mode)
        p.add_argument("--output-dir", required=True, type=Path)
        p.add_argument("--engine", choices=("graph", "direct"), default="graph")
        p.add_argument("--model-provider", choices=("ollama", "gemini"), default="ollama")
        p.add_argument("--model", required=True, help="Explicit model ID; configured externally")
        if mode == "live":
            p.add_argument("--question", required=True)
            p.add_argument("--companies", required=True, help="3-5 company names, comma-separated")
            p.add_argument("--max-results", type=int, default=3)
            p.add_argument("--tavily-keyless", action="store_true",
                           help="Explicitly opt into Tavily's limited keyless search mode")
        if mode == "extract-recorded":
            p.add_argument("--source-run-dir", required=True, type=Path,
                           help="Directory with a previously captured sources.json")
    oracle = sub.add_parser("oracle-fixture")
    oracle.add_argument("--output-dir", required=True, type=Path)
    oracle.add_argument("--engine", choices=("graph", "direct"), default="graph")
    replay = sub.add_parser("replay")
    replay.add_argument("--run-dir", required=True, type=Path)
    replay.add_argument("--output-dir", required=True, type=Path)
    replay.add_argument("--engine", choices=("graph", "direct"), default="graph")
    review = sub.add_parser("review")
    review.add_argument("--report", required=True, type=Path)
    review.add_argument("--labels", required=True, type=Path)
    review.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "replay":
        metrics = replay_run(args.run_dir, args.output_dir, engine=args.engine)
    elif args.command == "review":
        metrics = review_run(args.report, args.labels, args.output)
    elif args.command == "oracle-fixture":
        gold = load_snapshot()
        request = ResearchRequest.model_validate(gold["request"])
        metrics = qualify_run(request, SnapshotRetriever(gold), CuratedExtractor(gold),
                              args.output_dir, engine=args.engine, mode="human-curated-oracle", gold=gold)
    elif args.command == "extract-fixture":
        gold = load_snapshot()
        request = ResearchRequest.model_validate(gold["request"])
        metrics = qualify_run(request, SnapshotRetriever(gold), _make_model(args),
                              args.output_dir, engine=args.engine, mode="llm-on-curated-excerpts", gold=gold)
    elif args.command == "extract-recorded":
        capture = _read_json(args.source_run_dir / "sources.json")
        request = ResearchRequest.model_validate(capture["request"])
        sources = [Source.model_validate(s) for s in capture["sources"]]
        if capture.get("source_sha256") != {s.id: _digest(s.text) for s in sources}:
            raise ValueError("Recorded source text hash mismatch")
        metrics = qualify_run(request, _RecordedRetriever(request, sources), _make_model(args),
                              args.output_dir, engine=args.engine, mode="llm-on-recorded-sources")
    else:
        request = ResearchRequest(
            question=args.question,
            companies=[x.strip() for x in args.companies.split(",")],
        )
        retriever = TavilyRetriever(results_per_query=args.max_results, allow_keyless=args.tavily_keyless)
        metrics = qualify_run(request, retriever, _make_model(args),
                              args.output_dir, engine=args.engine, mode="live")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
