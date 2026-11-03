"""Offline example: python -m deepresearch_v2"""
from .core import ResearchRequest, markdown_table, run_research
from .example import COMPANIES, QUESTION, FixtureExtractor, FixtureRetriever

report = run_research(ResearchRequest(question=QUESTION, companies=COMPANIES),
                      FixtureRetriever(), FixtureExtractor())
print("SYNTHETIC OFFLINE FIXTURE — not real company intelligence\n")
print(markdown_table(report))
