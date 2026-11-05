# M2.5a — Externally labeled verdict evaluation (AVeriTeC)

## What this measures

Our prior M2.4 checks that a human-review packet is complete and consistent.
It does not measure automated claim understanding. This slice measures how
often a DeepSeek classifier agrees with the independent AVeriTeC gold verdict
when the benchmark's human-annotated question/answer evidence is given.

This is an ORACLE-EVIDENCE test: our system does NOT retrieve web evidence
or generate questions. The official AVeriTeC score requires both retrieval
quality and correct verdicts. It must not be reported for this experiment.
AVeriTeC's labels are Supported, Refuted, Not Enough Evidence, and
Conflicting Evidence/Cherrypicking. Development split: 500 examples.
Dataset license: CC BY-NC 4.0; noncommercial use only.

Official repository: https://github.com/MichSchli/AVeriTeC
Official dataset: https://fever.ai/dataset/averitec.html
Pinned dev JSON upstream revision:
https://github.com/MichSchli/AVeriTeC/blob/7c62d1ec8df3fb560d6efe2b85fa191135636f81/data/dev.json

No third-party benchmark examples are committed to our project. Download the
original dataset directly and keep it in Downloads or a separate local
directory, not under a Git-tracked path.

## Protocol

- Validate official dev JSON array and four gold labels (Pydantic).
- Deterministically balance a sample using seed 5243; at most 50 cases.
- Feed the model only the claim, speaker/date metadata and gold QA answer
  text. The gold label, justification, fact-check URL and source URLs are
  NOT part of the model prompt.
- DeepSeek Flash is called via its existing non-thinking structured-output
  configuration. Each call produces one four-way verdict, explanation and
  optional evidence IDs.
- Deterministic Python computes four-class accuracy, macro-F1, per-class
  precision, recall, F1, confusion matrix and an always-Supported baseline.
- Capture SHA-256 of dataset bytes, pinned source commit, sample IDs, seed,
  model and prompt versions, token use, per-case errors and partial failures.
- No chart generation or merged FinanceBench score.

## User-local instructions (when ready)

If Git transport still fails on the Mac, use your browser to download
(1) the pinned AVeriTeC dev.json and save as ~/Downloads/dev.json, and
(2) this feature branch ZIP. Safari may extract the ZIP automatically.

AVeriTeC data:
https://github.com/MichSchli/AVeriTeC/raw/7c62d1ec8df3fb560d6efe2b85fa191135636f81/data/dev.json

Feature ZIP:
https://github.com/WAWQAQMAKABAKA/DeepResearchReportGenerator/archive/refs/heads/feat/v2-m2p5a-averitec-verdict-benchmark.zip

From the EXTRACTED M2.5a directory:

    PY=~/DeepResearchReportGenerator/.venv/bin/python

First perform a no-charge schema/sample check:

    "$PY" -m deepresearch_v2.averitec_benchmark inspect --dataset ~/Downloads/dev.json --count 20

Then, after deciding to make 20 billed DeepSeek API requests:

    "$PY" -m deepresearch_v2.averitec_benchmark run --dataset ~/Downloads/dev.json --count 20 --model deepseek-flash --output-dir ~/DeepResearchReportGenerator/build/m2p5a-averitec-20

Finally, offline replay-scoring with the unchanged original data:

    "$PY" -m deepresearch_v2.averitec_benchmark replay-score --dataset ~/Downloads/dev.json --run-dir ~/DeepResearchReportGenerator/build/m2p5a-averitec-20

Do not paste DEEPSEEK_API_KEY into chat. Each run requires a NEW output
directory; failures leave progress records but cannot yet be resumed.
No claim that DeepSeek passed this benchmark is warranted until a live run.

## Acceptance criteria

- CI on Python 3.11 and 3.13 runs offline synthetic schema, stratification,
  leak prevention, scoring, input-checksum, failure and replay tests.
- User-local inspect on downloaded real AVeriTeC dev.json succeeds.
- User-local DeepSeek run produces nonempty true-benchmark per-class scores.
- Replay yields the same deterministic metrics and a list of failures.
- Our research project's quoted-claim validator remains unchanged.

## Next milestone

M2.5b: FinanceBench's expert-annotated open 150 financial questions and
evidence from filings. Different task and scoring protocol. Public sample
license CC BY-NC 4.0; no dataset redistribution.

Source: https://github.com/patronus-ai/financebench
