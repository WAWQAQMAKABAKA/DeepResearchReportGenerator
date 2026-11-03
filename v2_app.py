"""Thin Streamlit preview for the v2 runtime."""
import streamlit as st

from deepresearch_v2.core import ResearchRequest, markdown_table
from deepresearch_v2.example import COMPANIES, QUESTION, FixtureExtractor, FixtureRetriever
from deepresearch_v2.graph import build_graph

st.set_page_config(page_title="Deep Research v2 — M1", layout="wide")
st.title("Deep Research v2 — Evidence-linked comparison")
mode = st.selectbox("Mode", ["Offline synthetic fixture", "Live Tavily + LLM"])
question = st.text_input("Research question", value=QUESTION if mode.startswith("Offline") else "Compare AI coding agents")
company_text = st.text_input("Companies (comma-separated, 3–5)", value=", ".join(COMPANIES))
if mode.startswith("Live"):
    provider = st.selectbox("Model provider", ["ollama", "gemini"])
    model = st.text_input("Model name", value="llama3:8b" if provider == "ollama" else "gemini-2.0-flash")

if st.button("Run research", type="primary"):
    try:
        request = ResearchRequest(question=question, companies=[x.strip() for x in company_text.split(",")])
        if mode.startswith("Offline"):
            if request.companies != COMPANIES:
                st.error("The offline fixture supports only AlphaCode, BetaPilot, GammaForge. Choose Live for other companies.")
                st.stop()
            retriever, extractor = FixtureRetriever(), FixtureExtractor()
        else:
            from deepresearch_v2.live import LangChainExtractor, TavilyRetriever
            retriever, extractor = TavilyRetriever(), LangChainExtractor(provider, model)
        report = build_graph(retriever, extractor).invoke({"request": request})["report"]
        st.session_state["m1_report"] = report
        st.session_state["m1_report_mode"] = mode
    except Exception as exc:
        st.error(f"Research failed: {exc}")

if "m1_report" in st.session_state:
    report = st.session_state["m1_report"]
    if st.session_state.get("m1_report_mode", "").startswith("Offline"):
        st.warning("Synthetic fixture data; not real companies or market research.")
    st.markdown(markdown_table(report))
    st.caption(f"Sources: {len(report.sources)} · Accepted claims: {len(report.claims)} · Rejected candidates: {report.rejected_candidates}")
    with st.expander("Evidence ledger"):
        for claim in report.claims:
            ev = next(e for e in report.evidence if e.id == claim.evidence_ids[0])
            src = next(s for s in report.sources if s.id == ev.source_id)
            st.markdown(f"**{claim.company} / {claim.field}: {claim.value}** — [{src.id}]({src.url})")
            st.caption(f"Source-reported (not independently verified): “{ev.quote}”")
    st.download_button("Download validated report JSON", report.model_dump_json(indent=2), "v2-m1-report.json")
