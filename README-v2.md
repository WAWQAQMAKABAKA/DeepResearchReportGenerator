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

## M2.1 — Live-adapter qualification harness (backend only)

This adds **reproducible provider/model integration**, not yet a verified autonomous research result. The intended initial stack is Tavily search + LangChain/Ollama structured Pydantic extraction. The existing Gemini adapter can also be selected after installing its optional dependency (`langchain-google-genai`) and providing credentials. No Streamlit changes are required.

### Qualification stages

1. **Capture retrieval**: Execute two bounded queries per company (maximum five Tavily results per query), normalize sources, record query provenance, whether text came from raw content or a snippet, truncation, UTC retrieval time, and SHA-256 content digests in `sources.json`.
2. **Capture extraction**: Pass the captured source excerpts to the configured structured-output model, one call per company. Persist **every returned candidate** (including unsupported ones) to `candidates.json`, together with model identifier, prompt revision, invocation source IDs and response/usage metadata when available. The downstream validator still rejects broken quote/claim references.
3. **Compose**: Save `report.json`, `report.md`, `evaluation.json`, and `status.json`. Outputs explicitly identify model claims as **UNREVIEWED**; quotation/value matching is *not* semantic verification.
4. **Replay**: Validate source and candidate SHA-256 digests and rerun the same LangGraph composition **without network or model calls**. Byte-for-byte compare reports from the original run and replay.
5. **Human claim audit**: An independent reviewer labels each accepted claim as `supported`, `contradicted`, `insufficient`, `stale`, or `uncertain`. The separate review score reports review coverage and support rate, **not independent validation of vendor disclosures**.

### Dependency setup and demo (no API keys)

```bash
python -m pip install -r requirements-m2p1.txt
python -m pytest -q tests/test_m1.py tests/test_headless.py tests/test_public_benchmark.py tests/test_qualification.py

# Human-curated oracle fixture. NOT an LLM test.
python -m deepresearch_v2.qualification oracle-fixture --engine graph --output-dir build/m2p1-oracle
python -m deepresearch_v2.qualification replay --engine graph --run-dir build/m2p1-oracle --output-dir build/m2p1-oracle-replay
cmp build/m2p1-oracle/report.json build/m2p1-oracle-replay/report.json
```

The fixture result is for Cursor, GitHub Copilot, and Devin, using the **short captured official-page excerpts** in `fixtures/ai_coding_agents_2026-11-03.json`. `gold_exact_alignment` is an exact-string diagnostic, not semantic precision/recall. This stage tests wiring, persistence and traceability only.

### Run a real model on the fixed reviewed corpus (one provider only)

Set up your locally accessible Ollama server and pull a structured-output-capable model, for example `ollama pull qwen2.5:7b`. Then run:

```bash
python -m deepresearch_v2.qualification extract-fixture \
  --engine graph --model-provider ollama --model qwen2.5:7b \
  --output-dir build/m2p1-llm-fixture
```

This isolates **model extraction quality** from search variability. It compares accepted model claims against the human-curated source/value baseline by normalized company/field/value triplet. Different correct paraphrases may score as nonmatches, and exact matches may still be unsupported: human passage review remains necessary.

### Bounded live search + extraction (requires connectivity)

Use an environment variable for Tavily credentials; do not put API keys in a commit, CLI argument, snapshot or report. A Tavily keyless mode exists in newer SDKs but has strict rate limits; using it requires explicit `--tavily-keyless` instead of an API key.

```bash
export TAVILY_API_KEY='...set locally, never commit...'
python -m deepresearch_v2.qualification live \
  --question 'Compare AI coding agents from public official sources' \
  --companies 'Cursor,GitHub Copilot,Devin' \
  --model-provider ollama --model qwen2.5:7b \
  --max-results 3 --output-dir build/m2p1-live-001

python -m deepresearch_v2.qualification replay \
  --run-dir build/m2p1-live-001 \
  --output-dir build/m2p1-live-001-replayed
```

To retry only the model on **previously captured sources** without paying for more Tavily calls:

```bash
python -m deepresearch_v2.qualification extract-recorded \
  --source-run-dir build/m2p1-live-001 \
  --model-provider ollama --model qwen2.5:7b \
  --output-dir build/m2p1-reextracted-001
```

`--output-dir` must not already exist, preventing accidental overwrite. If model extraction fails after retrieval, `sources.json` remains available and `status.json` records a failed stage without storing exception details or credentials.

### Manual review (use actual IDs from your own run)

Create a local `review-labels.json` such as:

```json
{"reviewer":"local-reviewer", "labels":[
  {"claim_id":"C1", "verdict":"supported", "note":"Directly stated in cited passage"},
  {"claim_id":"C2", "verdict":"stale", "note":"Older plan superseded by newer source"}
]}
```

```bash
python -m deepresearch_v2.qualification review \
  --report build/m2p1-live-001/report.json \
  --labels review-labels.json --output build/m2p1-live-001/review.json
```

A **complete** claim-support review requires labels for every accepted claim. `review.json` explicitly counts supported, contradicted, insufficient, stale and uncertain decisions. Any unreviewed/contradicted/stale claim prevents calling the report reliable. Source-reported claims are still not independently verified financial/business facts.

### Limits / acceptance gate

- Automated CI tests use **mock Tavily transport and mock structured-model responses**, alongside the pinned manually curated corpus. They cannot establish a successful live Tavily API request, a deployed Ollama server, or a production model's extraction accuracy.
- The current validator is **lexical**: it verifies that a value and quote appear in source text but can incorrectly accept a negated or outdated claim. A regression test deliberately documents this defect; manual labels and later semantic validation must address it.
- The stored text is a bounded excerpt (up to 6,000 characters per retrieved result, and only the first 4,500 characters per source sent to the model); this is **not** a complete crawl or legal archive. For third-party pages, review copyright/retention policy and keep live snapshots private unless their redistribution is permitted. `build/` outputs should not be committed.
- No claims of model accuracy, retrieval relevance, coverage improvement, financial figures, tool cost or UI readiness are made without measurement.

**M2.1 exit gate:** at least one genuine Tavily + configured-model run, reproducible snapshot replay, and a manually labeled claim set demonstrating acceptable support. Until then, adapter contract verification is complete but live quality qualification remains pending.
