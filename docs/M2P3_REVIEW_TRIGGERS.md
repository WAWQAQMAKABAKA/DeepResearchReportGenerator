# M2.3 — Conservative review-trigger sidecar

## Goal and strict non-goals

Based on the user's **successful live M2.2 run** (17/17 lexically accepted,
10/24 table fields, 7/9 exact curated value-triplet matches), add a small,
inspectable quality flag layer that surfaces **known semantic and temporal
risks for human review** without changing M2.2 extraction or claim acceptance.

Do **not** call these rules automatic semantic verification, inferred current
pricing, or determination of stale evidence. There is no general published_at
field in M2.2 `Source`. The `retrieved_at` field means *retrieved*, not
*published*, and must never be substituted for publication time.

## Observations motivating the rules

The user pasted 17 values and evidence excerpts (C1–C17). We reconstruct a
**text-only test fixture**, `fixtures/m2p2_transcribed_candidates_2026-11-04.json`,
from that pasted terminal transcript. This fixture is **not** the raw API
capture, contains no credentials, and must not be described as a verbatim
provider trace.

- **C8**: "all paid Copilot subscribers" is an access/eligibility population
  rather than proof of a target customer segment.
- **C10**: "For everyday coding with agents" is a use-case slogan, not a
  concrete customer segment.
- **C6, C9, C15, C16**: Price-only values omit tier, effective date, or enough
  commercial context for cross-product comparisons. A quoted number still
  needs contextual review.
- **C16/C17**: Their source is explicitly titled historical in the pinned
  corpus. This is an editorial metadata cue, **not a robust stale-date
  detector**. C17 can be a sound historically attributed segment; C16 is not
  fit as current pricing without independent verification.

## Implementation contract

- New pure `deepresearch_v2/quality_flags.py` with Pydantic models:
  `ReviewTrigger` and `ReviewTriggerReport`.
- Four conservative regex/title triggers:
  `pricing_amount_without_tier`,
  `eligibility_as_target_customer`,
  `use_case_as_target_customer`,
  `historical_source_title`.
- New `quality_flags.json` sidecar saved on **all** qualification runs and
  offline replays, with rule version and source-linked claim IDs.
- Additional `evaluation.json` diagnostic fields:
  `claims_with_review_triggers`, `review_trigger_counts`,
  `review_triggers_are_verdicts=false`.
- No modification to `Claim` or `Source` contracts, `compose_report`,
  the M2.2 extraction prompt, or Markdown rendering. In particular, a
  flagged claim remains `UNREVIEWED` and lexically accepted.

These rules are **intentionally not comprehensive**: unknown stale prices,
difficult misclassifications, and negation/context errors can pass without
a flag. Unflagged does NOT mean verified or publication-ready. Rates on this
single frozen corpus are regression targets, not empirical generalization.

## Acceptance tests and run commands

CI checks that the reconstructed 17 candidates remain 17/17 lexical accepts,
with 7 distinct flagged claims and 8 total flags. It checks the oracle remains
9/9 accepted, that tiered values are not flagged for missing tier, and that
quality_flags.json is byte-identical in headless graph replay.

After getting the M2.3 code (via Git or a browser ZIP), the original live
M2.2 run remains untouched. Use the **M2.3 code** and reuse the existing
local Python virtual environment:

```bash
~/DeepResearchReportGenerator/.venv/bin/python -m deepresearch_v2.qualification replay \
  --engine graph \
  --run-dir ~/DeepResearchReportGenerator/build/m2p2-deepseek-exact-spans-001 \
  --output-dir ~/DeepResearchReportGenerator/build/m2p2-deepseek-exact-spans-001-m2p3-replay
```

This call is offline (no DeepSeek charges). Inspect
`quality_flags.json` and `evaluation.json`, and compare replayed
`report.json` to the original. An existing output dir is not overwritten.
If Git is still blocked by an HTTP/2 transport issue, download a ZIP of
the feature branch to a **separate** folder and run from its root.

## Follow-up gate (not implemented here)

Independently review the flagged and unflagged claims. Then design a
structured `plan_tier`, `effective_as_of` and `source_published_at` contract
using actual published provenance, rather than brittle natural-language
heuristics. Whether to suppress flagged claims from publication is a separate
product/policy decision; do not quietly suppress or rewrite source claims.
