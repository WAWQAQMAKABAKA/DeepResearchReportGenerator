# M2.4 — Explicit review packet and comparison-readiness gate

## Why this is necessary

The user's genuine M2.3 replay of 17 DeepSeek-extracted claims identified
7 claims with 8 risk flags while preserving the original 17/17 lexical accepts.
The old `review` CLI only records *passage support*, not whether a claim is in
the correct comparison field or temporally relevant. Source `retrieved_at`
is not proof of `published_at`.

M2.4 implements a separate, fully offline workflow:
1. Generate a **review packet** containing every claim, quote, source URL,
   retrieval timestamp, and the M2.3 review triggers.
2. Generate a **blank review manifest**, bound by SHA-256 to the exact
   `report.json` bytes; no verdicts default to `supported`.
3. The reviewer supplies `passage_support`, `field_fit`,
   `temporal_relevance`, `ready_for_comparison`, rationale,
   and explicit trigger acknowledgements.
4. Generate a **gate report**, reporting either `blocked` or
   `review_complete`; it never generates a publishable report.

**This is a consistency gate, not a factuality validator**. User-supplied
reviewer IDs are not independently authenticated; test reviewers and
assistant-assisted judgments are not automatically human sign-off.
Flag acknowledgements do not establish independently verified dates.

## Usage (offline, no API key needed)

Use M2.4 code from the `feat/v2-m2p4-review-gate` branch (browser ZIP is
fine if Terminal still has Git HTTP errors) with the pre-existing venv.

```bash
cd ~/Downloads/DeepResearchReportGenerator-feat-v2-m2p4-review-gate
PY=~/DeepResearchReportGenerator/.venv/bin/python
RUN=~/DeepResearchReportGenerator/build/m2p2-deepseek-exact-spans-001
OUT=~/DeepResearchReportGenerator/build/m2p4-review

"$PY" -m deepresearch_v2.qualification review-packet \
  --report "$RUN/report.json" \
  --packet "$OUT/review_packet.json" \
  --template "$OUT/review_template.json"

# Shows BLOCKED without filling in fake review decisions.
"$PY" -m deepresearch_v2.qualification review-gate \
  --report "$RUN/report.json" \
  --labels "$OUT/review_template.json" \
  --output "$OUT/gate-unreviewed.json"
```

Edit a **copy** of `review_template.json` with genuinely reviewed decisions
and a traceable rationale. Never edit captured `report.json`. For each
decision, all fields must be non-null and `note` non-empty. If
`ready_for_comparison=true` then the passage must be supported, field fit
appropriate, temporal relevance either current or not time sensitive, and
all associated trigger codes explicitly acknowledged.

If the full review is submitted, run `review-gate` again with a **new**
output path. The gate reports the IDs eligible for further report curation
but does not silently remove other claims or create a final report.

## Acceptance criteria

- 17 original claims preserved; no API calls or claim re-extraction.
- Packet includes source IDs, quotes, titles, URLs and M2.3 flags.
- Empty template is blocked, and an incomplete set of decisions is blocked.
- A claim cannot be ready if its support, field fit or temporal status fails.
- Flagged claim requires acknowledgement when marked ready.
- Reviewer input is bound to the *exact report bytes*; mismatch fails.
- No report is published by this milestone.
- Earlier `review` CLI remains backward compatible.

Test labels in CI are **synthetic** and must not be reported as human audit.
