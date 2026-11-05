# M2.5b: FinanceBench business QA evaluation (gold filing excerpts)

## Purpose

AVeriTeC M2.5a: 20 real DeepSeek Flash examples, 65% four-way
verdict accuracy, macro-F1 0.554; offline score replay confirmed.
That result does not validate financial or company research answers.

This separate M2.5b module evaluates financial question answering using the
open 150-case FinanceBench expert-annotated sample from public filings.
The full 10,231-question corpus is not publicly available. The open dataset
is for noncommercial use under CC BY-NC 4.0. No benchmark data is committed.

Sources:
- https://github.com/patronus-ai/financebench
- https://www.patronus.ai/research/financebench
- https://huggingface.co/datasets/PatronusAI/financebench
- Pinned GitHub commit cc39aeb4afdf33909ee1412188bf89035950c2eb

## Source inspection and task definition

Verified original JSONL format: 150 questions, 50 each of metrics-generated,
domain-relevant and novel-generated types. Every entry supplies expert answer,
justification and one or more evidence excerpts. The actual evidence object
uses the field name doc_name, while parts of the upstream README call it
evidence_doc_name. The adapter supports both, with doc_name preferred.
The largest evidence text total is below 16,000 characters for the original
open-source sample.

Give DeepSeek the question, company, document identifier, and expert-annotated
filing excerpt text, but NEVER give the model gold answer or justification.
The model returns a structured answer, rationale, and excerpt IDs. No PDF
retrieval, financial data scraping, or open-web research happens here.

## Deliberately conservative deterministic scoring

Business answers include numerical amounts, narrative explanations,
qualitative judgments and calculated ratios. This module does NOT use
exact-string matches as a supposed ground-truth accuracy score.

It records:
- normalized_exact_string_agreement: a surface-form diagnostic only
- strict_scalar_diagnostic: compare Decimal values when BOTH answers consist
  of standalone scalars with the same currency or percent marker
- requires_semantic_review: case IDs whose responses cannot be judged by
  the simple deterministic rules
- official_financebench_score is null; error_count_unverified is true

Even for numerical answers, different units, scaling or rounding need
context-specific adjudication. The module does not manufacture human grades,
infer missing units or declare paraphrases wrong. Deterministic arithmetic
belongs in a future business-specific evaluator with explicit answer schemas.

## Controls and reproducibility

- Verified, SHA-256-pinned local source dataset
- Repeatable sample by source question type with seed 5243, max 30 cases
- Default small run: 12 examples (four per type)
- DeepSeek Flash with non-thinking function calling, same as M2.5a
- Exact model/prompt versions and per-example token usage
- No silent truncation of evidence tables; strict 16k char budget
- Immutable output dir, partial-run failure status
- Offline replay-score checks input checksum and sample IDs

## Mac usage

If your Terminal Git transport is unreliable, use Safari to download:
1. Data file:
https://raw.githubusercontent.com/patronus-ai/financebench/cc39aeb4afdf33909ee1412188bf89035950c2eb/data/financebench_open_source.jsonl

Save as ~/Downloads/financebench_open_source.jsonl. If Safari displays text,
use File > Save As and select plain source rather than a web archive.

2. M2.5b feature branch ZIP:
https://github.com/WAWQAQMAKABAKA/DeepResearchReportGenerator/archive/refs/heads/feat/v2-m2p5b-financebench-oracle-qa.zip

After ZIP extraction, enter the extracted project directory and invoke the
existing Python virtual environment. No new virtual environment is required.

Free real-dataset inspection:

    ~/DeepResearchReportGenerator/.venv/bin/python -m deepresearch_v2.financebench_benchmark inspect --dataset ~/Downloads/financebench_open_source.jsonl --count 12

Optional first DeepSeek run (makes 12 paid API requests):

    ~/DeepResearchReportGenerator/.venv/bin/python -m deepresearch_v2.financebench_benchmark run --dataset ~/Downloads/financebench_open_source.jsonl --count 12 --model deepseek-flash --output-dir ~/DeepResearchReportGenerator/build/m2p5b-financebench-12

Free deterministic replay scoring:

    ~/DeepResearchReportGenerator/.venv/bin/python -m deepresearch_v2.financebench_benchmark replay-score --dataset ~/Downloads/financebench_open_source.jsonl --run-dir ~/DeepResearchReportGenerator/build/m2p5b-financebench-12

Never paste API keys into chat. A run cannot overwrite an existing folder.

## Gates

- CI Python 3.11/3.13 tests use synthetic samples ONLY
- Real 150-row FinanceBench file must be validated locally
- Run paid inference only after the user chooses to do so
- No numerical or semantic accuracy claim from these diagnostics
- Keep FinanceBench separate from AVeriTeC four-way scores
- No changes to the fork main or original shared Columbia main

Next research question: define a defensible semi-automatic adjudicator for
numeric/qualitative FinanceBench cases or use the existing published
human-correctness annotations of prior model responses where applicable,
without misrepresenting those as grades for new DeepSeek responses.
