# M2.7 — Evidence-status preview integrated into company research

## Goal

Connect the existing M1–M2.6 evidence and validation work to the actual
Technology Company Competitive Landscape research workflow. Focus on a
genuinely useful preview and publication gate rather than another standalone
benchmark classifier.

The underlying company report uses ResearchReport, Source, Evidence, Claim
and CompanyProfile. Its accepted claims have VERBATIM source excerpts,
but not independently established semantic entailment. The M2.3 review
trigger heuristics and M2.4 explicit review-consistency gate exist already.
The separate FinanceBench and AVeriTeC results are evaluation datasets,
NOT company claim authorizations.

## What this milestone does

The capture and replay pipeline automatically writes two additive artifacts:

- evidence_preview.json (structured, source-traceable status register)
- evidence_preview.html (static read-only HTML, source links and quotes)

For every requested company and all eight comparison fields:

1. No lexically accepted claim -> not_established
2. Different reported values -> unresolved_conflict
3. One consistent source-reported value, no completed review ->
   source_reported_unreviewed
4. Complete and consistent review with user-declared ready decisions ->
   review_declared_ready
5. Complete review excluding all claims in a slot ->
   review_declared_excluded
6. Mixed readiness within same field -> mixed_review_decisions

The HTML uses static HTML escaping for every untrusted company, source,
model and URL display string; no JavaScript or generated executable code.
It links to captured source URLs and displays verbatim evidence, retrieval
timestamps, and review trigger codes. Retrieval time is NOT the source's
publication or effective date.

The report JSON's existing comparison table is not trusted as independent
evidence. Preview fields are recomposed from validated claim/evidence
relationships and the source excerpt map. All original outputs remain
unchanged apart from these new additive preview files.

No calculations are inferred from unstructured model statements. There
are currently no structured ReportCalculation/MetricObservation records
produced by the company report pipeline. Consequently the preview states:

    calculated_observations = []
    calculation_status = not_supported_by_current_report_schema

This is preferable to inventing business numbers or transplanting
FinanceBench oracle formula evidence into a different company report.

## Publication and review safety

- Every output sets publication_ready=false and
  semantic_support_automatically_verified=false.
- A report with no review manifest remains unreviewed, even with no flags.
- An incomplete or blocked M2.4 manifest cannot mark any claim ready.
- For a completed review, the label is review_declared_ready (not
  independently_supported). M2.4 reviews are user-supplied and reviewer
  IDs are not independently authenticated.
- Conflicting values stay unresolved even when all decisions are marked
  ready.
- Wrong SHA-256 in the review manifest raises an error before output
  creation.
- No UI or exporter in this milestone publishes a report.

## Execution paths

An existing completed or new qualification run now automatically contains:

    build/<qualification-run>/evidence_preview.json
    build/<qualification-run>/evidence_preview.html

For the previously created M2.2 run, do NOT recreate the DeepSeek API
results. Use the offline CLI below to build a new preview:

    cd ~/Downloads/DeepResearchReportGenerator-feat-v2-m2p7-evidence-preview
    ~/DeepResearchReportGenerator/.venv/bin/python \
      -m deepresearch_v2.qualification report-preview \
      --report ~/DeepResearchReportGenerator/build/m2p2-deepseek-exact-spans-001/report.json \
      --output-dir ~/DeepResearchReportGenerator/build/m2p7-company-preview-001

Open the saved HTML locally:

    open ~/DeepResearchReportGenerator/build/m2p7-company-preview-001/evidence_preview.html

Optional: append --review-labels with an independently supplied M2.4
ReviewManifest JSON, verified by SHA-256 to the exact report bytes.
A blank/incomplete manifest remains blocked; do NOT fabricate a reviewer
or review decisions to make it pass.

You may also use the module's own standalone CLI:

    python -m deepresearch_v2.evidence_preview --report path/report.json --output-dir new/path

In all cases the output directory must be fresh and the original
report.json is never modified.

Branch ZIP:
https://github.com/WAWQAQMAKABAKA/DeepResearchReportGenerator/archive/refs/heads/feat/v2-m2p7-evidence-preview.zip

## Explicit acceptance gates

- CI Python 3.11 and 3.13 passes all legacy plus eight M2.7 tests.
- Test actual pinned public excerpts from three coding-agent companies
  through both direct Python capture and LangGraph replay; confirm the
  preview register is identical and only additive.
- Synthetic adversarial tests cover source/claim HTML injection attempts,
  multi-source contradictions, missing fields, incomplete or stale review,
  synthetic complete review marked user-declared only, immutable outputs,
  and successful offline post-run CLI.
- User-local preview of the real earlier DeepSeek company research run
  still pending. Do not count synthetic review decisions as human review.
- No official AVeriTeC/FinanceBench accuracy improvement asserted.
- The Columbia repository and both main branches remain unchanged.

## Next smallest slice

Once the actual 17-claim saved DeepSeek report preview is inspected, define
a separate typed MetricObservation/CalculationSpec contract with explicit
period, units, operands and source references, ONLY if the collected company
dataset contains suitable comparable numeric observations. Evaluate that
contract on disjoint company research cases, not the same 12 FinanceBench
questions used to design arithmetic warnings.
