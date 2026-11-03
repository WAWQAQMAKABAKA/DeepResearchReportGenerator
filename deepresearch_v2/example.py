"""Synthetic fixtures only. Company names and statements are fictional."""
from datetime import datetime, timezone
from .core import Candidate, ExtractionBatch, ResearchRequest, Source

QUESTION = "Compare fictional AI coding agent products"
COMPANIES = ["AlphaCode", "BetaPilot", "GammaForge"]


class FixtureRetriever:
    def retrieve(self, request: ResearchRequest) -> list[Source]:
        facts = [
            ("AlphaCode", "AlphaCode offers an editor assistant for software teams. AlphaCode lists a free tier."),
            ("BetaPilot", "BetaPilot offers terminal coding tools for developers. BetaPilot lists seat-based pricing."),
            ("GammaForge", "GammaForge describes a repository analysis assistant for engineering teams."),
        ]
        return [Source(id=f"S{i}", company=company, url=f"https://fixture.example/{i}",
                       title=f"Synthetic {company} fixture", text=text,
                       retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc), provider="fixture")
                for i, (company, text) in enumerate(facts, 1)]


class FixtureExtractor:
    def extract(self, request: ResearchRequest, sources: list[Source]) -> ExtractionBatch:
        return ExtractionBatch(items=[
            Candidate(company="AlphaCode", field="product", value="Editor assistant",
                      source_id="S1", quote="AlphaCode offers an editor assistant for software teams."),
            Candidate(company="AlphaCode", field="pricing", value="free tier",
                      source_id="S1", quote="AlphaCode lists a free tier."),
            Candidate(company="BetaPilot", field="product", value="Terminal coding tools",
                      source_id="S2", quote="BetaPilot offers terminal coding tools for developers."),
            Candidate(company="BetaPilot", field="pricing", value="Seat-based",
                      source_id="S2", quote="BetaPilot lists seat-based pricing."),
            Candidate(company="GammaForge", field="product", value="Repository analysis assistant",
                      source_id="S3", quote="GammaForge describes a repository analysis assistant for engineering teams."),
        ])
