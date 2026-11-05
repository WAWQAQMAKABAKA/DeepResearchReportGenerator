# M2.6.1 — Bounded arithmetic provenance (offline)

## Goal

Reduce misleading "number not found literally in evidence" signals in our
M2.6 FinanceBench audit. A numerical *result* may be computed from values
in an annotated company filing. An absence of the final result as a literal
does NOT mean it is unsupported.

This milestone introduces a read-only, deterministic, **advisory-only**
audit. The previous M2.5b predictions and M2.6 audit remain byte-identical;
a fresh M2.6.1 sidecar is created. No new model calls or human labels.

## Inputs and provenance contract

The auditor consumes:
- Pinned external FinanceBench dataset with the original 150 annotated
  company filing questions and evidence snippets, CC BY-NC 4.0.
- Existing completed DeepSeek Flash M2.5b run (12 examples by default).
- Original M2.6 business_audit.json from the same run.

It replays the original score and checks exact dataset SHA-256, captured
manifest/prediction/evaluation SHA-256, parent audit schema, every selected
case, and original M2.6 audit output. A mismatch fails closed before
creating a new output directory.

Sidecar fields include:
- source and parent SHA-256 values;
- provenance_kind: numeric_literal_candidate, calculated_candidate,
  unresolved_scale, unresolved_calculation, unresolved, or nonnumeric;
- formula identifier, operands found/not found in **cited** excerpts,
  computed expected number and arithmetic consistency;
- unchanged numeric reference-agreement status and original warning codes;
- numeric_semantic_support = unverified;
- metric_definition_correct = unverified;
- publication_ready = false;
- overall_business_accuracy = null.

## Whitelisted arithmetic shapes

Four formulas are recognized only when the question contains the relevant
task, the model's *recorded* rationale contains the bounded expression,
and the model has cited excerpts for the operand values:

| Formula | Arithmetic | Requested task |
|---|---|---|
| revenue_yoy | (current - prior) / prior * 100 | revenue year-on-year percentage |
| inventory_turnover | COGS / ((current inventory + prior inventory) / 2) | inventory turnover |
| eps_guidance_delta | new EPS growth guidance - old EPS growth guidance | percentage-point guidance change |
| dividend_payout_ratio | cash dividends paid / shareholder net income | dividend payout ratio |

Validation uses Python Decimal and the precision of the model's displayed
answer. An extra 1% relative tolerance is allowed ONLY when the question
explicitly asks "roughly" or "approximately", and results are separately
labeled approximate_with_roughly_qualifier. That is NOT the same as exact
agreement.

A calculation must have all operands present as numeric tokens in model-
cited source excerpts. If a rationale is fabricated, incomplete, wrong
or unsupported by cited values, the answer is unresolved. No arbitrary
LLM-provided Python or mathematical code is executed.

Answers containing multiple numbers are never graded as one scalar.
Standalone dollar amounts are not automatically trusted without a
verified source currency scale. Literal matches are candidate
observations, not financial facts.

## Known limits

This is **not semantic source validation**. Merely locating an operand
anywhere in the excerpt does not prove that it is attached to the correct
metric, column, reporting period, currency scale, or financial denominator.

The formula is detected in the model's existing rationale and may itself
be unsupported. This implementation does not extract authoritative
year/line-item mappings from filings. No retrieval, financial metric
ontology, or arbitrary formula engine is implemented.

A nonnumeric answer (Adobe's "Yes", for instance) is marked nonnumeric,
not wrong. A source note containing "411" without a unit may be
unresolved_scale rather than an automatic direct fact.

The original M2.6 warnings remain visible for traceability. A calculated
candidate helps interpret a prior literal-not-found warning; it does not
erase that warning or upgrade publication readiness.

## Tests and acceptance

- Existing M2.6 and all prior regressions must remain green in CI
  on Python 3.11 and Python 3.13.
- New adversarial regression cases use **disjoint synthetic values**,
  not just the 12 previously viewed FinanceBench cases.
- Test false operand claims, calculation mismatches, missing citations,
  multiple numbers, amount scale without currency confirmation, and
  altered dataset/parent audit checksums.
- User-local test runs on the *unchanged* 12-case real FinanceBench
  experiment and original M2.6 audit with zero new DeepSeek API calls.
- No new accuracy or benchmark improvement is claimed on either the
  original 12 cases or synthetic fixtures.

## Mac: free real-run audit

Download the M2.6.1 source branch ZIP through Safari:

https://github.com/WAWQAQMAKABAKA/DeepResearchReportGenerator/archive/refs/heads/feat/v2-m2p6p1-arithmetic-provenance.zip

Save/retain the existing original dataset and completed run.
From the extracted M2.6.1 source directory:

    ~/DeepResearchReportGenerator/.venv/bin/python \
      -m deepresearch_v2.arithmetic_provenance \
      --dataset ~/Downloads/financebench_open_source.jsonl \
      --run-dir ~/DeepResearchReportGenerator/build/m2p5b-financebench-12 \
      --prior-audit-dir ~/DeepResearchReportGenerator/build/m2p6-business-audit-12 \
      --output-dir ~/DeepResearchReportGenerator/build/m2p6p1-provenance-12

This revalidates and reads the original run; output is written ONLY to
the new directory, as arithmetic_provenance.json. It never contacts
DeepSeek.

## Next gate

Inspect case-level formula classes and unresolved warnings. For a future
stronger validator, require source tables with explicit line-item and
period mappings before treating operand occurrence as true evidence
support. Run on disjoint real FinanceBench examples before reporting
any model quality improvements.
