from __future__ import annotations

import logging

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from src.agent.graph import run_agent
from src.config import get_settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

HALLUCINATION_QUERY = "what is requirement for level 2 engineer ?"


def _build_ragas_llm_and_embeddings():
    """Return (llm, embeddings) using RAGAS-native providers, or None pair to fall back to OpenAI."""
    settings = get_settings()
    try:
        from ollama import Client
        from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
        from ragas.llms import llm_factory

        llm = llm_factory(
            model=settings.ollama_model_name,
            client=Client(host=settings.ollama_base_url),
        )
        embeddings = RagasHFEmbeddings(model=settings.embedding_model_name)
        logger.info("RAGAS will use local Ollama LLM + HuggingFace embeddings.")
        return llm, embeddings
    except Exception as exc:
        logger.warning(
            "Ollama wiring failed (%s); falling back to OpenAI for RAGAS.", exc
        )
        return None, None


def run_edge_case_bench() -> pd.DataFrame:
    # ── 1. Run the agent against the known failing query ──────────────────────
    logger.info("Running agent for edge-case query: %r", HALLUCINATION_QUERY)
    state = run_agent(HALLUCINATION_QUERY)

    generation: str = state.get("generation", "") or ""
    contexts: list[str] = [
        doc.page_content
        for doc in state.get("documents", [])
        if hasattr(doc, "page_content")
    ]

    logger.info("Agent returned %d grounded chunk(s).", len(contexts))
    logger.info("Generation preview: %.200s", generation)

    # ── 2. Build the HuggingFace Dataset RAGAS expects ────────────────────────
    hf_dataset = Dataset.from_dict(
        {
            "question": [HALLUCINATION_QUERY],
            "answer": [generation],
            "contexts": [contexts],
        }
    )

    # ── 3. Wire LLM / embeddings ──────────────────────────────────────────────
    llm, embeddings = _build_ragas_llm_and_embeddings()

    eval_kwargs: dict = dict(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy],
    )
    if llm is not None:
        eval_kwargs["llm"] = llm
    if embeddings is not None:
        eval_kwargs["embeddings"] = embeddings

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    logger.info("Running RAGAS evaluation …")
    result = evaluate(**eval_kwargs)

    # ── 5. Output as Markdown table ───────────────────────────────────────────
    df: pd.DataFrame = (
        result.to_pandas() if hasattr(result, "to_pandas") else pd.DataFrame([result])
    )

    print("\n=== RAGAS Edge-Case Hallucination Report ===")
    print(f"Query : {HALLUCINATION_QUERY!r}")
    print(f"Chunks: {len(contexts)}")
    print()
    print(df.to_markdown(index=False))
    return df


def main() -> None:
    run_edge_case_bench()


if __name__ == "__main__":
    main()
