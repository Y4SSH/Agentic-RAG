# Agentic RAG Knowledge System

A production-grade, self-correcting Retrieval-Augmented Generation (RAG) system built with modern LangChain and LangGraph orchestration.

Designed for reliability, observability, and local execution, this system ingests PDFs into a persistent vector store and runs a reproducible retrieve → grade → generate → hallucinate-check workflow. It exposes a fully traceable chat UI via Streamlit, allowing operators to validate every mathematical and agentic decision hop.

---

## Key Features

* **Deterministic Anti-Hallucination Guardrails:** Employs an `empty_response` node to hard-stop the generator when relevance grading returns zero documents, completely eliminating zero-context hallucinations.
* **Strict Vector Thresholding:** Uses normalized cosine similarity (`retrieval_score_threshold = 0.4`) rather than L2 distance to strictly filter out cross-document contamination before chunks ever reach the LLM.
* **Single-Chunk Retrieval (`k=1`):** Prevents intra-document blending ("context smoothies") in smaller models by forcing extreme extraction precision.
* **Defensive Fallback Mechanisms:** Every LLM call is wrapped in network timeouts and error handlers. If the LLM host goes offline, the system gracefully degrades to a deterministic token-overlap heuristic and extractive snippet generation.
* **Complete Observability:** The Streamlit UI strips out raw LLM noise and presents a clean, hop-by-hop execution trace (e.g., *“Retrieved 1 chunk(s) [max 1, threshold score > 0.4] (0 discarded)”*) alongside the exact source chunks used.
* **Quantitative Evaluation:** Includes a built-in `ragas_bench.py` harness to mathematically evaluate *Faithfulness* and *Answer Relevancy*.

---

## System Architecture

1. **User-Facing Layer (`app.py`):** Streamlit dashboard featuring chat input, dynamic assistant responses, a PDF upload sidebar, and a collapsible execution trace diagnostic panel.
2. **Orchestration Layer (`graph.py`):** A LangGraph state machine routing the workflow between retrieval, grading, rewriting, generation, and hallucination checks.
3. **Logic Nodes (`nodes.py`):** The intelligence layer utilizing strict, binary-choice few-shot `PromptTemplates` to force small local models to act as ruthless evaluators.
4. **Ingestion Pipeline (`chunk_and_embed.py`):** Discovers PDFs, chunks them via `RecursiveCharacterTextSplitter` (size 500, overlap 50), embeds them using `all-MiniLM-L6-v2`, and persists them to a local Chroma DB.
5. **LLM Host:** Designed for local execution via the Ollama daemon (optimized for `tinyllama` testing, scalable to `llama3.1`).

---

## Getting Started

### Prerequisites

* Python 3.14+ (or compatible 3.10+ version)
* Ollama installed and in your PATH

### Installation

Clone the repository and set up your virtual environment:

```powershell
# Create and activate virtual environment (Windows)
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

```

### Start the Local LLM (Ollama)

Ensure the Ollama daemon is running (`http://localhost:11434`) and pull the default lightweight model:

```powershell
ollama pull tinyllama

```

### Run the Application

Launch the Streamlit UI:

```powershell
python -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501

```

---

## 🧪 Testing & Evaluation

### Automated Smoke Tests

Run the built-in sanity checks to validate the `empty_response` fallback and basic grounded generation:

```powershell
.venv\Scripts\python.exe scripts\run_smoke_tests.py

```

*(Exit code 0 indicates all state machine routing and fallbacks are healthy).*

### RAGAS Benchmarking

To mathematically evaluate the system against known hallucination or cross-contamination edge cases:

```powershell
python -m src.eval.ragas_bench

```

This generates a Pandas DataFrame output scoring the system's *Faithfulness* and *Answer Relevancy*.

---

## Project Structure

```text
├── .env                        # Environment variables (OLLAMA_MODEL_NAME=tinyllama)
├── requirements.txt            # Pinned dependencies (langchain, langgraph, chromadb, etc.)
├── app.py                      # Streamlit UI entry point
├── chroma_db/                  # Local persistent vector store
├── data/                       # Directory for source PDFs
├── scripts/
│   ├── run_smoke_tests.py      # Automated local testing
│   └── cross_validate_windows.ps1 # Windows environment validation helper
└── src/
    ├── config.py               # Pydantic BaseSettings, thresholds, & paths
    ├── agent/
    │   ├── graph.py            # LangGraph routing and state machine
    │   ├── nodes.py            # Retrieval, grading, and generation logic
    │   └── state.py            # TypedDict GraphState definition
    ├── eval/
    │   └── ragas_bench.py      # Quantitative evaluation harness
    └── ingestion/
        └── chunk_and_embed.py  # PDF processing and vector persistence

```

---

##  Known Limitations

* **RAM Constraints:** While `tinyllama` (~637 MB) runs smoothly on lower-end hardware, swapping to `llama3.1` or `phi3` requires upwards of 4GB+ of dedicated RAM.
* **Windows Paths:** When interacting via PowerShell, ensure file paths are carefully quoted if they contain spaces.