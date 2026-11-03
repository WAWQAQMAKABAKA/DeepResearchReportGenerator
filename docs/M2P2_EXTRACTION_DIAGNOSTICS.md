# M2.2 — Extraction contract and rejection diagnostics

## Observed live baseline (2026-11-03)

Against eight pre-captured official-source excerpts for Cursor, GitHub Copilot
and Devin, DeepSeek Flash completed its first non-thinking structured-output
run. It returned 12 candidates: 2 accepted, 10 rejected. The user's local
forensic script counted **6 value_not_in_quote** (paraphrases rather than
literal value spans) and **4 unknown_company** ("Cognition Devin" instead of
the requested "Devin"). The two accepted claims remain **UNREVIEWED**.
There were **0 exact company/field/value triplet matches** against the
manually curated reference; this is a lexical diagnostic, not accuracy.

## Smallest corrective change

1. Enforce the **exact study company label** in the system and user prompts.
   Do not silently map vendor/product aliases in the validator.
2. Instruct the model that `quote` must be copied verbatim from the given
   source, and `value` must be a **contiguous verbatim substring of quote**.
   Paraphrases and multi-fact syntheses must be omitted at this stage.
3. Centralize existing fail-closed composition checks as
   `candidate_rejection_reason()`; add deterministic
   `candidate_diagnostics.json` (one outcome/reason per candidate) and
   `rejection_breakdown` in `evaluation.json`. No auto-acceptance of
   previously rejected candidates.
4. Bump the extraction prompt version to
   `company-extract-m2.2-exact-spans-v1` for reproducible comparison.

The M2.2 **offline tests** reproduce the observed 6+4 breakdown using
synthetic candidates and test replay. This is *not* a replay of the user's
original proprietary API response, because the actual `-003` candidate
snapshot remains on their Mac.

## Reproduce local baseline without paying for more API calls

```bash
cd ~/DeepResearchReportGenerator
# Switch to M2.2 feature branch after safeguarding any uncommitted work.
git fetch origin
git switch feat/v2-m2p2-extraction-contract
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements-deepseek.txt

python -m deepresearch_v2.qualification replay \
  --engine graph \
  --run-dir build/m2p1-deepseek-flash-003 \
  --output-dir build/m2p1-deepseek-flash-003-replayed
```

The replay produces diagnostics for the old 12 model candidates without an API
call. Expected rejection breakdown (from user diagnostic): 6 value_not_in_quote,
4 unknown_company, 2 accepted. Do not overwrite the `-003` originals.

## Run M2.2 live experiment on the same frozen excerpts

```bash
python -m deepresearch_v2.qualification extract-fixture \
  --engine graph \
  --model-provider deepseek \
  --model deepseek-flash \
  --output-dir build/m2p2-deepseek-exact-spans-001
```

Compare `evaluation.json`, `candidate_diagnostics.json`, and model usage
metadata against the prior baseline. **Do not** claim that the prompt fixed the
issue until a real model invocation confirms it.

## Acceptance gate

No model-error API response; same 8-source corpus; diagnostic counts
self-consistent; no broadened or weakened lexical acceptance; all accepted
claims independently reviewed for passage support and staleness. Improvement
requires measured results, not merely a higher accepted count.

This remains a **headless** milestone. No UI or framework migration.
