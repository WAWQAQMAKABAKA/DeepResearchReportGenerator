# Deep Research v2 — M1.1 headless validation

**Current capability:** A deterministic evidence-linked three-company comparison. The bundled `AlphaCode`, `BetaPilot`, and `GammaForge` are **fictional fixtures**, not facts about real companies. Live retrieval/extraction adapters exist but are **not yet qualified**.

## Architecture

- `deepresearch_v2/core.py`: Pydantic `Source`, `Evidence`, `Claim`, `CompanyProfile`, `MetricObservation`, `ReportComponent` contracts; candidate validation; deterministic table.
- `deepresearch_v2/graph.py`: `retrieve → extract → validate_and_compose` using LangGraph.
- `deepresearch_v2/cli.py`: headless reproducible fixture runner, JSON and Markdown output.
- `deepresearch_v2/live.py`: optional Tavily/Ollama/Gemini adapters (not used for this milestone).
- `deepresearch_v2/example.py`: synthetic, deterministic sources and extraction.
- `v2_app.py`: earlier optional Streamlit preview; UI integration deliberately deferred.

## M1.1 quick start (no UI, no network API keys)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-m1.txt
python -m pytest -q tests/test_m1.py tests/test_headless.py
python -m deepresearch_v2 --engine graph --output-dir build/m1-fixture
```

The final command must run the real LangGraph workflow and creates:

- `build/m1-fixture/report.json`: typed profiles, source text, evidence passages, claims, report components.
- `build/m1-fixture/report.md`: Markdown table with source URLs, missing values, and a synthetic-data disclaimer.

For a **core-only** fallback when LangGraph is not installed, use:

```bash
python -m deepresearch_v2 --engine direct --output-dir build/m1-direct
```

Direct mode does **not** establish LangGraph compatibility. GitHub Actions (`.github/workflows/m1-headless.yml`) is intended to install LangGraph, run the graph integration tests on Python 3.11 and 3.13, and compare generated file bytes across repeated runs. A workflow file alone is not proof that a CI run succeeded; inspect the PR checks.

## Scope and evidence caveats

The first milestone is a functioning typed comparison, **not** a complete evidence-grounded industry report. The validator checks exact quote membership in stored source text and lexical value membership in that quote; it cannot establish semantic entailment or source credibility. Incompatible supported values remain visible as multiple reports. Unknown fields stay `Not established`. Do not interpret synthetic metrics or examples as current intelligence.

The original shared repository `shayantist/STAT5243-DeepResearch` remains unchanged. Changes belong to `WAWQAQMAKABAKA/DeepResearchReportGenerator` feature branch.

## Later work (not part of M1.1)

Once headless LangGraph is reproducible, verify a bounded live retrieval/extraction run with three actual companies, snapshot sources for replay, and evaluate claim-to-evidence semantic support. Only then extend the report with a sourced timeline and narrative, and subsequently build a UI.


## M2 — First public-source excerpt replay (3 real AI coding agents)

This is a **bounded public-data experiment**, not live model extraction. Official webpages were manually inspected via web retrieval on 2026-11-03; this environment does not permit direct HTTP retrieval and has no Tavily/model API credentials. To keep the experiment independently replayable, `fixtures/ai_coding_agents_2026-11-03.json` stores eight short source passages and nine manually curated candidate claims. Each passage has an official URL, a captured date, and a source-published date where known. `deepresearch_v2/public_benchmark.py` runs these through the unchanged production validator/composer via the actual LangGraph pipeline.

```bash
python -m pip install -r requirements-m1.txt
python -m pytest -q tests/test_m1.py tests/test_headless.py tests/test_public_benchmark.py
python -m deepresearch_v2.public_benchmark --engine graph --output-dir build/m2-public-snapshot
```

Generated outputs: `report.json` (eight Source records, nine evidence-linked Claim records, three CompanyProfiles), `report.md` (table), and `evaluation.json` (8/9/24 counts, source text digests, qualification flags). The CI workflow `.github/workflows/m2-public-replay.yml` checks Python 3.11/3.13 and byte-for-byte deterministic outputs. **All source documents are first-party but their claims may be vendor assertions, not independently verified facts.** The historical December 2024 Devin $500/month article is preserved as evidence of prior pricing, not used as current pricing because the official April 2026 announcement describes new plans. The current price entries are for different tiers/usage bundles and are **not comparable unit costs**.

### Boundaries and what is still needed

The extraction fixture is human curated (an *oracle*), and the replay does not test Tavily retrieval, LLM extraction accuracy, fresh HTTP fetching, or semantic entailment. Test `test_known_limitation_lexical_validator_not_semantic_entailment` intentionally proves that the current lexical validator accepts a statement contradicted by a negation in the same quote. Therefore a green CI run does **not** establish real-world agent reliability. Next gate: configure one approved retrieval/model integration, save the complete fetched source passages with timestamps and hashes (under an appropriate retention/copyright policy), run bounded model extraction, and manually label evidence entailment on the resulting claims before extending the report/UI.
