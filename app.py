from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.agent.graph import run_agent
from src.config import get_settings
from src.ingestion.chunk_and_embed import ingest_directory

settings = get_settings()


@st.cache_resource(show_spinner=False)
def _cached_app_title() -> str:
    return settings.streamlit_page_title


st.set_page_config(
    page_title=_cached_app_title(),
    page_icon=settings.streamlit_page_icon,
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #e6eef8;
        --muted: #94a3b8;
        --line: rgba(255, 255, 255, 0.06);
        --surface: rgba(6, 10, 14, 0.6);
    }
    body {
        background-color: #0b1220;
        color: var(--ink);
    }
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 1.5rem;
        max-width: 1400px;
    }
    [data-testid="stChatInput"] {
        border-top: 1px solid rgba(255,255,255,0.04);
        padding-top: 0.75rem;
        margin-top: 0.5rem;
    }
    [data-testid="stChatInput"] textarea {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        box-shadow: none !important;
        background: rgba(12, 18, 28, 0.75) !important;
        color: var(--ink) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.12) !important;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(6, 10, 14, 0.96), rgba(11, 18, 32, 0.96));
        color: var(--ink);
        border: 1px solid rgba(255, 255, 255, 0.02);
        border-radius: 24px;
        padding: 1.5rem 1.75rem;
        box-shadow: 0 18px 48px rgba(3, 6, 9, 0.6);
        margin-bottom: 1rem;
    }
    .hero-card h1 {
        font-size: 2rem;
        margin-bottom: 0.25rem;
    }
    .hero-card p {
        margin: 0;
        opacity: 0.9;
    }
    .trace-summary {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.2rem 0 0.75rem 0;
    }
    .metric-pill {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.6);
        color: var(--ink);
        font-weight: 600;
    }
    .trace-step {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #7c3aed;
        border-right: 1px solid rgba(255,255,255,0.02);
        border-top: 1px solid rgba(255,255,255,0.02);
        border-bottom: 1px solid rgba(255,255,255,0.02);
        border-radius: 0 12px 12px 0;
        padding: 0.5rem 0.7rem;
        margin: 0.35rem 0;
        color: var(--ink);
        line-height: 1.35;
    }
    .compact-status {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.5rem;
        margin: 0.35rem 0 0.75rem 0;
    }
    .status-chip {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255,255,255,0.02);
        border-radius: 14px;
        padding: 0.55rem 0.7rem;
        color: var(--ink);
        font-size: 0.92rem;
        line-height: 1.25;
    }
    .status-chip strong {
        display: block;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
        margin-bottom: 0.2rem;
    }
    .source-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255,255,255,0.02);
        border-radius: 14px;
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.65rem;
        color: var(--ink);
    }
    .source-meta {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1>Agentic RAG Knowledge System</h1>
        <p>Upload PDFs, re-index the persistent vector store, and inspect a fully traceable LangGraph execution path.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _save_uploaded_pdfs(uploaded_files: list[Any]) -> list[Path]:
    saved_paths: list[Path] = []
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        target_path = settings.data_dir / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path)
    return saved_paths


def _render_execution_trace(state: dict[str, Any]) -> None:
    steps_taken = list(state.get("steps_taken", []))
    search_loop_count = int(state.get("search_loop_count", 0))
    documents = list(state.get("documents", []))

    st.markdown(
        f"""
        <div class="compact-status">
            <div class="status-chip"><strong>Search loops</strong>{search_loop_count}</div>
            <div class="status-chip"><strong>Grounded chunks</strong>{len(documents)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Execution Trace")
    with st.expander("Open architectural hop-by-hop trace", expanded=False):
        if not steps_taken:
            st.info("No execution trace available yet.")
        else:
            for index, step in enumerate(steps_taken, start=1):
                st.markdown(
                    f'<div class="trace-step"><strong>{index}.</strong> {step}</div>',
                    unsafe_allow_html=True,
                )


def _render_source_chunks(state: dict[str, Any]) -> None:
    documents = list(state.get("documents", []))
    with st.expander("Source chunks used in the last answer"):
        if not documents:
            st.info("No source chunks were retained after relevance grading.")
        else:
            for index, document in enumerate(documents, start=1):
                source = document.metadata.get("source", "unknown-source")
                page = document.metadata.get("page", "unknown-page")
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-meta">Chunk {index} • `{source}` • page `{page}`</div>
                        <div>{document.page_content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


st.sidebar.header("Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Saved files are written to the local data directory and immediately re-indexed into Chroma.",
)

if "last_upload_signature" not in st.session_state:
    st.session_state.last_upload_signature = None

if uploaded_files:
    upload_signature = tuple((file.name, file.size) for file in uploaded_files)
    if st.session_state.last_upload_signature != upload_signature:
        with st.spinner("Saving PDFs and rebuilding the vector index..."):
            saved_paths = _save_uploaded_pdfs(uploaded_files)
            summary = ingest_directory(settings.data_dir)
        st.session_state.last_upload_signature = upload_signature
        st.sidebar.success(
            f"Saved {len(saved_paths)} PDF(s) and re-indexed the knowledge base."
        )
        st.sidebar.caption(
            f"Indexed documents: {summary.documents_read} | chunks: {summary.chunks_created}"
        )
        st.sidebar.write("Saved files:")
        for path in saved_paths:
            st.sidebar.write(f"- {path.name}")

with st.sidebar.expander("Runtime configuration", expanded=True):
    sidebar_col1, sidebar_col2 = st.columns(2)
    sidebar_col1.metric("Chunk Size", settings.chunk_size)
    sidebar_col2.metric("Overlap", settings.chunk_overlap)
    st.caption(f"Collection: `{settings.collection_name}`")
    st.caption(f"Search loop limit: `{settings.search_loop_limit}`")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_state" not in st.session_state:
    st.session_state.last_state = None

trace_placeholder = st.empty()
sources_placeholder = st.empty()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the ingested knowledge base")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Running the LangGraph agent..."):
            state = run_agent(question)
            st.session_state.last_state = state
        st.markdown(state.get("generation", ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": state.get("generation", "")}
    )

if st.session_state.last_state:
    with st.sidebar.expander("Latest run diagnostics", expanded=False):
        _render_execution_trace(st.session_state.last_state)
        _render_source_chunks(st.session_state.last_state)
