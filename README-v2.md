# Deep Research v2 — Milestone 1

This milestone produces an evidence-linked **company comparison table**. It does not yet produce an executive summary, event timeline, or quantitative charts.

- `deepresearch_v2/core.py`: Pydantic contracts, quote validation, source/claim references, deterministic table.
- `deepresearch_v2/live.py`: Tavily and configurable Ollama/Gemini structured extraction (optional).
- `deepresearch_v2/graph.py`: LangGraph `retrieve → extract → validate_and_compose`.
- `v2_app.py`: thin Streamlit preview.
- `deepresearch_v2/example.py`: **fictional** offline fixture (not market intelligence).

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-v2.txt
python -m pytest -q tests/test_m1.py
python -m deepresearch_v2  # deterministic offline demonstration without Streamlit
streamlit run v2_app.py
```

Offline core tests and `python -m deepresearch_v2` need only `pydantic` and `pytest`; the graph and UI need `langgraph` and `streamlit` respectively. The Streamlit demo uses the graph in both modes.

For live mode, set `TAVILY_API_KEY`, configure a local Ollama model (default `llama3:8b`) or `GOOGLE_API_KEY` for Gemini, then select **Live Tavily + LLM** in Streamlit. Search snippets/page text are not independently verified. A candidate passes only if its quotation is found in the stored source text. That check is necessary but **not a semantic entailment guarantee**. Human review and claim-support evaluation are future work.

No live keys, real company metrics, timeline inference, or chart observations are included in offline fixtures. Invalid citations are rejected, source conflicts are presented as unresolved multiple reports, and unsupported cells say `Not established`.

The shared `shayantist/STAT5243-DeepResearch` repository remains unchanged. This v2 code lives only in the user-owned fork's feature branch.
