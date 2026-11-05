# M2.6 — Deterministic FinanceBench business-answer audit

## Purpose and current evidence

We now have real, immutable DeepSeek-Flash benchmark runs:
- AVeriTeC 20 gold-QA examples: accuracy 65%, macro-F1 0.554, 0/5
  recall on Conflicting Evidence/Cherrypicking. Offline score replay passed.
- FinanceBench 12 gold filing-excerpt questions: 1/12 exact-string agreements,
  2/2 equal among two strictly comparable numerical outputs, and 10 cases
  requiring other evaluation. The latter numbers are NOT business QA accuracy.

AI-assisted inspection of the 12 original FinanceBench cases found eight clear
evidence-backed matches, one conceptual metric problem (AmEx's pretax margin
treated as an operating-margin proxy), and three source/reference caveats:
3M PP&E value $8.738bn in source versus $8.70bn expert answer; PepsiCo $411M
from a financial statement note even though question asks for amounts directly
stated in an income statement; JPMorgan TBVPS $66.56 is not established
liquidation proceeds. These are investigator judgments, not an externally
human-verified grading dataset.

M2.6 adds an OFFLINE, read-only audit over the captured original FinanceBench
predictions. It is not a new LLM judge, a financial decision engine, a
publication approval, or full semantic claim entailment.

## Scope

New module: deepresearch_v2/business_validation.py

* Exact-source provenance: verify the unchanged FinanceBench dataset SHA,
  selected sample IDs, completed recorded run, and re-score its predictions.
  Refuse if saved evaluation differs. Capture SHA-256 of the original manifest,
  predictions and evaluation files in the new audit output.
* Numeric diagnostics: parse standalone scalars or exactly one explicit
  unit-bearing number; normalize USD scale in millions/billions when clearly
  instructed in the question; compare using Decimal (no binary-float drift).
  Distinguish exact, within half of the *reference* last displayed digit,
  different, and not-comparable. Never silently accept $8.738bn as $8.70bn:
  the difference of $38M exceeds $5M half-step at two decimal places.
* Source-text occurrence: inspect numeric tokens in the model-cited excerpts
  and identify simple literal/explicit-unit-scale matches. This is a
  **weak diagnostic**, NOT proof that the value corresponds to the right
  financial metric, year, company, period, or supported reasoning. Calculated
  metrics can legitimately have no literal numeric match.
* Conservative warning-only heuristics:
  - pretax income substituted as an operating-margin proxy
  - liquidation proceeds inferred from book value / TBVPS
  - a financial statement note used when the question specifically asks
    for amounts directly in the income statement
  - discrepancy against reference number
  - an answer requiring nonnumeric or semantic assessment
* Exactly the source-cited document pages (document identifier and 0-indexed
  page index), raw answer, expert reference and rationale-linked warning codes
  are in the audit. No automatic claim is declared correct for publication.

Each case contains:
- numeric.status
- answer_numeric_literal_in_cited_evidence
- advisory flags and evidence page refs
- semantic_correctness = unverified
- publication_ready = false

The sidecar has overall_business_accuracy = null and
independently_human_validated = false. This is deliberate.

## What the audit cannot establish

A matching number does not prove its *meaning*, the proper denominator,
accounting scope, correct reporting period, or industry relevance. Percent
change or ratio correctness may require deterministic calculations from
multiple input values with a validated formula; we have not implemented a
general financial formula registry. Free-text paraphrase equivalence is not
decided by this module. It does not test financial document retrieval or
external source freshness.

The known 12-case example analysis is used to design transparent failure
triggers, but NOT to train or tune the DeepSeek prompt and NOT counted as an
independent improvement dataset.

## Acceptance and replay

- New synthetic pytest tests for units, precision, multi-number refusal,
  source literal lookup, financial interpretation warnings, tampering,
  immutable sidecars and fail-closed source SHA.
- Existing CI regression must continue to pass Python 3.11/3.13.
- User-local invocation against the immutable actual captured FinanceBench
  12-case run, with the *same dataset*, produces the audit sidecar.
- The existing FinanceBench evaluation.json and predictions.jsonl must be
  byte-identical before and after the audit.
- Do not report a new numerical business accuracy or claim human review
  based only on these diagnostics.

## Free local audit (NO model/API calls)

Download this feature ZIP through Safari if GitHub HTTPS via Terminal fails:
https://github.com/WAWQAQMAKABAKA/DeepResearchReportGenerator/archive/refs/heads/feat/v2-m2p6-business-answer-audit.zip

From the extracted ZIP root, reuse existing Python venv:

    PY=~/DeepResearchReportGenerator/.venv/bin/python
    "$PY" -m deepresearch_v2.business_validation \
      --dataset ~/Downloads/financebench_open_source.jsonl \
      --run-dir ~/DeepResearchReportGenerator/build/m2p5b-financebench-12 \
      --output-dir ~/DeepResearchReportGenerator/build/m2p6-business-audit-12

The command does NOT contact DeepSeek and does not rewrite the original run.
It prints summary counts; the complete 12-case audit is in
build/m2p6-business-audit-12/business_audit.json. Output directory must
not exist.

## Subsequent evidence-based milestone

Only after examining this output should we specify a small, separately
evaluated financial formula/citation verifier. Potential tests: explicit
capital-expenditure and margin denominators with validated formulas,
classification of intended metric, and open questions with multiple possible
interpretations. Preserve a disjoint evaluation set; do not reuse these
12 cases to claim improvement without fresh samples.
