"""LangGraph orchestration; same pure composition step as offline mode."""
from __future__ import annotations
from typing import TypedDict

from .core import ExtractionBatch, ResearchReport, ResearchRequest, Source, compose_report


class ResearchState(TypedDict, total=False):
    request: ResearchRequest
    sources: list[Source]
    extracted: ExtractionBatch
    report: ResearchReport


def build_graph(retriever, extractor):
    from langgraph.graph import END, START, StateGraph

    workflow = StateGraph(ResearchState)
    workflow.add_node("retrieve", lambda state: {"sources": retriever.retrieve(state["request"])})
    workflow.add_node("extract", lambda state: {
        "extracted": extractor.extract(state["request"], state["sources"])
    })
    workflow.add_node("validate_and_compose", lambda state: {
        "report": compose_report(state["request"], state["sources"], state["extracted"])
    })
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "extract")
    workflow.add_edge("extract", "validate_and_compose")
    workflow.add_edge("validate_and_compose", END)
    return workflow.compile()
