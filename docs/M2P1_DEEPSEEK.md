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


## DeepSeek 400: thinking mode vs forced Pydantic tools

The default `deepseek-flash` mode is **thinking enabled**. A LangChain
`with_structured_output(method="function_calling")` schema forces a named tool;
DeepSeek's Chat Completions API rejects that with HTTP 400 and the message
`Thinking mode does not support this tool_choice`. The DeepSeek adapter now
explicitly passes `extra_body={"thinking": {"type": "disabled"}}` when using
Pydantic extraction. This keeps forced tool selection, instead of changing to
unreliable `tool_choice=auto`. Only DeepSeek is affected; Ollama/Gemini paths
remain unchanged.

If an earlier attempt failed, the output directory was intentionally created
and `status.json` should say `failed`; do **not** reuse that directory.
After `git pull --ff-only`, choose a fresh directory such as
`build/m2p1-deepseek-flash-002`. Verify the existing shell exports
`DEEPSEEK_API_KEY` without displaying it:

```bash
[ -n "$DEEPSEEK_API_KEY" ] && echo "DeepSeek key configured" || echo "Key missing"
```

The CI regression tests verify this *client configuration*. They do not make
a real DeepSeek API request. A successful user-run extraction and subsequent
human review remain necessary for qualification.
