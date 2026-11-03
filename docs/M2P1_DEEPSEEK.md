# DeepSeek API extraction for Deep Research v2 (M2.1)

**Decision:** Use the existing LangGraph + Pydantic evidence pipeline, and plug in the DeepSeek cloud API as an LLM provider. DeepSeek Harness (DSH) is a separate plugin-based agent runtime: evaluate it later only against a concrete orchestration need, not as a prerequisite for the first extraction-quality measurement.

The current DeepSeek API docs recommend `deepseek-flash` for the lightweight model. Do not start new experiments on the deprecated `deepseek-chat` alias. The adapter calls `ChatDeepSeek.with_structured_output(ExtractionBatch, method="function_calling", include_raw=True)` and still passes outputs through our evidence validator.

## Mac setup (from your existing clone)

```bash
cd ~/DeepResearchReportGenerator
git fetch origin
git switch feat/v2-m2p1-deepseek-provider
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements-deepseek.txt

# DeepSeek Harness' own config may not export a key to the Python process.
# Don't paste the key into chat or put it in Git.
printf 'DeepSeek API key (hidden input): '
read -rs DEEPSEEK_API_KEY
printf '\n'
export DEEPSEEK_API_KEY

python -m deepresearch_v2.qualification extract-fixture \
  --engine graph \
  --model-provider deepseek \
  --model deepseek-flash \
  --output-dir build/m2p1-deepseek-flash-001
```

The extraction fixture contains saved public-source excerpts for Cursor, GitHub Copilot and Devin. It makes approximately three cloud requests, one per company; Tavily is *not* needed for this first test.

Inspect `build/m2p1-deepseek-flash-001/evaluation.json`, `candidates.json`, and `report.md`. The status is **UNREVIEWED_CLAIMS** even when lexical checks pass. Reconstruct without another model request:

```bash
python -m deepresearch_v2.qualification replay \
  --engine graph \
  --run-dir build/m2p1-deepseek-flash-001 \
  --output-dir build/m2p1-deepseek-flash-001-replay
```

If your existing working tree contains uncommitted work, inspect `git status` before switching branches. The output directory must be new each time. No API key is written to JSON artifacts.

## Next acceptance test

A genuine run on your machine with your existing DeepSeek key, plus manually reviewed claim-to-passage support labels. CI tests only import/instantiate the real SDK and exercise **mocked responses**; they do not call DeepSeek's billable endpoint or demonstrate semantic reliability. For full research later, the existing `live` subcommand also requires `TAVILY_API_KEY` (or explicitly enabled keyless mode).
