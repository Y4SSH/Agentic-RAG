from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any

from httpx import ConnectError, ReadTimeout, TimeoutException
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

_MODEL_CONNECTION_ERRORS = (
    ConnectError,
    ReadTimeout,
    TimeoutException,
    OSError,
    ValueError,
)


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=_settings().embedding_model_name)


@lru_cache(maxsize=1)
def _vectorstore() -> Chroma:
    settings = _settings()
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=_embeddings(),
    )


@lru_cache(maxsize=1)
def _chat_model() -> Any:
    settings = _settings()
    if settings.has_openai_key:
        return ChatOpenAI(
            model=settings.openai_model_name,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key.get_secret_value(),
        )

    ollama_base_url = settings.ollama_base_url.strip() or "http://localhost:11434"
    if not ollama_base_url.startswith(("http://", "https://")):
        ollama_base_url = f"http://{ollama_base_url}"

    return ChatOllama(
        model=settings.ollama_model_name.strip(),
        base_url=ollama_base_url,
        temperature=settings.openai_temperature,
    )


def _message_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
    return str(message)


def _append_step(state: dict[str, Any], step: str) -> list[str]:
    steps = list(state.get("steps_taken", []))
    steps.append(step)
    return steps


def _document_preview(document: Document, max_chars: int = 1600) -> str:
    content = document.page_content.replace("\r", " ").strip()
    if len(content) > max_chars:
        content = f"{content[:max_chars].rstrip()}..."
    source = document.metadata.get("source", "unknown-source")
    page = document.metadata.get("page", "unknown-page")
    return f"Source: {source} | Page: {page}\n{content}"


def _build_binary_chain(system_text: str, human_text: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human", human_text),
        ]
    )
    return prompt | _chat_model() | StrOutputParser()


def _normalize_binary_response(raw_text: str) -> str:
    return raw_text.strip().lower()


def _is_yes_response(raw_text: str) -> bool:
    return "yes" in _normalize_binary_response(raw_text)


def _invoke_binary_decision(
    system_text: str,
    human_text: str,
    variables: dict[str, Any],
    error_label: str,
) -> tuple[bool, str, bool]:
    chain = _build_binary_chain(system_text, human_text)
    try:
        raw_response = chain.invoke(variables)
    except _MODEL_CONNECTION_ERRORS:
        logger.exception("%s failure", error_label)
        return False, "", True
    except Exception:
        logger.exception("%s failure", error_label)
        return False, "", True

    normalized = _normalize_binary_response(_message_to_text(raw_response))
    return _is_yes_response(normalized), normalized, False


def _heuristic_relevance(question: str, content: str) -> tuple[bool, str]:
    question_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    content_tokens = set(re.findall(r"[a-z0-9]+", content.lower()))
    if not question_tokens or not content_tokens:
        return False, "Heuristic overlap score 0.00"
    overlap = len(question_tokens & content_tokens) / max(len(question_tokens), 1)
    relevant = overlap >= 0.12
    reason = f"Heuristic overlap score {overlap:.2f}"
    return relevant, reason


def _extractive_answer(question: str, documents: list[Document]) -> str:
    if not documents:
        return (
            "I could not find grounded source documents for this question in the current vector store. "
            "Please add relevant PDFs or refine the question."
        )

    snippets = []
    for index, document in enumerate(documents[:3], start=1):
        source = document.metadata.get("source", "unknown-source")
        page = document.metadata.get("page", "unknown-page")
        content = document.page_content.strip().replace("\r", " ")
        snippets.append(f"{index}. {source} (page {page}): {content[:400]}")
    return (
        f"Grounded answer for question: {question}\n"
        f"The best supporting evidence currently available is:\n" + "\n".join(snippets)
    )


def _safety_notice(state: dict[str, Any]) -> str:
    settings = _settings()
    steps_taken = list(state.get("steps_taken", []))
    search_loop_count = int(state.get("search_loop_count", 0))
    hallucination_failures = sum(
        1
        for step in steps_taken
        if step.lower().startswith("hallucination check failed")
    )
    rewrite_failures = sum(
        1 for step in steps_taken if "query rewrite fallback used after" in step.lower()
    )
    generation_failures = sum(
        1 for step in steps_taken if "generation fallback used after" in step.lower()
    )

    if hallucination_failures and rewrite_failures == 0:
        return (
            "Fallback activated after repeated grounding failures. "
            f"Rewrite count={search_loop_count}, hallucination failures={hallucination_failures}, "
            f"generation fallbacks={generation_failures}. "
            "Please start Ollama or provide a working model endpoint."
        )

    return (
        "Fallback activated after repeated retrieval and rewrite attempts. "
        f"The safety limit of {settings.search_loop_limit} query rewrites was reached (count={search_loop_count}). "
        "Please narrow the question or add more source documents."
    )


def empty_response(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation": "I'm sorry, but the provided documents do not contain any information to answer your question.",
        "steps_taken": state["steps_taken"] + ["Triggered empty response fallback"],
    }


def retrieve(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"].strip()
    settings = _settings()
    steps = _append_step(state, f"Retrieving chunks for query: {question}")
    try:
        if not question:
            return {"documents": [], "steps_taken": steps}

        # similarity_search_with_score returns (Document, float) where the score
        # is raw L2 distance: lower = more similar. Chroma's relevance normalisation
        # is unreliable with langchain_community, so we gate on distance directly.
        # For all-MiniLM-L6-v2: distance < 1.3 is a safe relevance cutoff.
        # k=1 enforces a single-chunk context window to prevent intra-document blending.
        scored: list[tuple[Document, float]] = (
            _vectorstore().similarity_search_with_score(
                question, k=1
            )
        )

        documents: list[Document] = []
        discarded: int = 0
        for doc, distance in scored:
            if distance < settings.retrieval_score_threshold:
                documents.append(doc)
            else:
                discarded += 1
                logger.debug(
                    "Chunk discarded (L2=%.4f >= threshold=%.2f): %s",
                    distance,
                    settings.retrieval_score_threshold,
                    doc.metadata.get("source", "?"),
                )

        steps[-1] = (
            f"Retrieved {len(documents)} chunk(s) [max 1, threshold L2 < "
            f"{settings.retrieval_score_threshold}] ({discarded} discarded) "
            f"for query: {question}"
        )
        return {"documents": documents, "steps_taken": steps}
    except Exception as exc:
        logger.exception("Retrieval failure")
        steps.append(f"Retrieval failed: {exc}")
        return {"documents": [], "steps_taken": steps}


def grade_documents(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"]
    documents = list(state.get("documents", []))
    graded_documents: list[Document] = []

    if not documents:
        steps = _append_step(
            state,
            "Graded chunks: 0 relevant / 0 total; no documents available for review",
        )
        return {"documents": [], "steps_taken": steps}

    prompt = PromptTemplate(
        template=(
            """You are a strict, merciless grader evaluating if a document is relevant to a user's question. 

    Here are the strict rules:
    - If the document does not explicitly contain the entity, concept, or answer to the question, you MUST say 'no'.
    - Do not be helpful. If in doubt, say 'no'.

    Example 1:
    Document: The company revenue in Q3 was $50M.
    Question: What was the Q4 revenue?
    Grade: no

    Example 2:
    Document: The Mars colony uses 45 liters of oxygen.
    Question: Who is Peter Griffin?
    Grade: no

    Example 3:
    Document: The hydroponics base salary is 85,000 credits.
    Question: How much does a hydroponics tech make?
    Grade: yes

    Now, grade the following:
    Document: \n\n {context} \n\n
    Question: {question}
    Grade (output exactly one word, 'yes' or 'no'):"""
        ),
        input_variables=["context", "question"],
    )

    for index, document in enumerate(documents, start=1):
        prompt_variables = {
            "question": question,
            "context": _document_preview(document),
        }
        try:
            chain = prompt | _chat_model() | StrOutputParser()
            raw_response = chain.invoke(prompt_variables)
            normalized = _normalize_binary_response(_message_to_text(raw_response))
            used_fallback = False
            if not normalized:
                # treat empty model output as fallback
                used_fallback = True
        except _MODEL_CONNECTION_ERRORS:
            logger.exception("Document grading failure")
            normalized = ""
            used_fallback = True
        except Exception:
            logger.exception("Document grading failure")
            normalized = ""
            used_fallback = True

        if used_fallback or not normalized:
            relevant, reason = _heuristic_relevance(question, document.page_content)
            reason = f"{reason}; model unavailable"
        else:
            relevant = _is_yes_response(normalized)
            reason = f"Model decision: {normalized}"

        if relevant:
            graded_documents.append(document)

    steps = _append_step(
        state,
        f"Graded chunks: {len(graded_documents)} relevant / {len(documents)} total",
    )
    return {"documents": graded_documents, "steps_taken": steps}


def rewrite_query(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"].strip()
    search_loop_count = int(state.get("search_loop_count", 0)) + 1
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the user's query into a concise search-optimized keyword string. "
                    "Return only the rewritten query and nothing else.",
                ),
                (
                    "human",
                    "Original question: {question}\n\nRecent execution trace:\n{steps}",
                ),
            ]
        )
        chain = prompt | _chat_model() | StrOutputParser()
        response = chain.invoke(
            {"question": question, "steps": "\n".join(state.get("steps_taken", []))}
        )
        rewritten = _message_to_text(response).strip()
        rewritten = rewritten.splitlines()[0].strip().strip('"').strip("'")
        if not rewritten:
            raise ValueError("The query rewrite model returned an empty string.")
    except Exception as exc:
        logger.exception("Query rewrite failure")
        rewritten = re.sub(r"\s+", " ", question).strip()
        rewritten = re.sub(r"[?!.]+$", "", rewritten)
        rewritten = rewritten or question
        rewritten = f"{rewritten} retrieval"
        steps = _append_step(
            state, f"Query rewrite fallback used after {exc.__class__.__name__}"
        )
    else:
        steps = _append_step(state, f"Rewrote query to: {rewritten}")

    steps.append(f"Search loop count incremented to {search_loop_count}")
    return {
        "question": rewritten,
        "search_loop_count": search_loop_count,
        "steps_taken": steps,
    }


def generate(state: dict[str, Any]) -> dict[str, Any]:
    question = state["question"].strip()
    documents = list(state.get("documents", []))
    context_block = "\n\n".join(
        _document_preview(document, max_chars=1200) for document in documents
    )
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a production RAG generator. Answer only from the provided source context. "
                    "If the context is insufficient, say so explicitly and recommend what to add.",
                ),
                (
                    "human",
                    "Question: {question}\n\nSource context:\n{context}\n\nWrite a clean, direct answer.",
                ),
            ]
        )
        chain = prompt | _chat_model() | StrOutputParser()
        response = chain.invoke(
            {"question": question, "context": context_block or "No context available."}
        )
        answer = _message_to_text(response).strip()
        if not answer:
            raise ValueError("The generation model returned an empty response.")
    except Exception as exc:
        logger.exception("Generation failure; using extractive fallback")
        answer = _extractive_answer(question, documents)
        steps = _append_step(
            state, f"Generation fallback used after {exc.__class__.__name__}"
        )
    else:
        steps = _append_step(
            state, f"Generated answer from {len(documents)} grounded chunk(s)"
        )

    return {"generation": answer, "steps_taken": steps}


def hallucination_check(state: dict[str, Any]) -> dict[str, Any]:
    answer = state.get("generation", "").strip()
    documents = list(state.get("documents", []))
    context_block = "\n\n".join(
        _document_preview(document, max_chars=1000) for document in documents
    )

    if not documents:
        steps = _append_step(
            state, "Hallucination check failed: no grounded documents were available"
        )
        return {"steps_taken": steps}

    prompt = PromptTemplate(
        template="""You are a strict grading system checking if an LLM's generated answer is entirely grounded in the provided source documents. 

    Here are the strict rules:
    - If the answer contains ANY facts, names, or numbers not explicitly found in the documents, it is a hallucination. You MUST answer 'yes' (it is hallucinating).
    - If the answer is completely supported by the documents, you MUST answer 'no' (it is NOT hallucinating).
    - Respond with exactly one word: 'yes' or 'no'.

    Example 1 (Hallucination):
    Documents: The base salary is 85,000 credits.
    Answer: The base salary is 85,000 credits and the CEO is Elon Musk.
    Grade: yes

    Example 2 (Grounded):
    Documents: A questionnaire is a structured set of written questions given to respondents.
    Answer: Questionnaires are written questions given to people to collect data.
    Grade: no

    Now, grade the following:
    Documents: \n\n {documents} \n\n
    Answer: {generation}
    Grade (output exactly one word, 'yes' or 'no'):""",
        input_variables=["documents", "generation"],
    )

    try:
        chain = prompt | _chat_model() | StrOutputParser()
        raw_response = chain.invoke(
            {
                "documents": context_block or "No context available.",
                "generation": answer,
            }
        )
        normalized = _normalize_binary_response(_message_to_text(raw_response))
        supported = _is_yes_response(normalized)
        if not normalized:
            supported = False
    except Exception:
        logger.exception("Hallucination check failure; using conservative fallback")
        supported = False

    if supported:
        steps = _append_step(state, "Hallucination check passed: Answer is grounded.")
    else:
        steps = _append_step(state, "Hallucination check failed: Regenerating draft.")

    return {"steps_taken": steps}


def fallback_notice(state: dict[str, Any]) -> dict[str, Any]:
    notice = _safety_notice(state)
    steps = _append_step(state, f"Fallback notice rendered: {notice}")
    return {"generation": notice, "steps_taken": steps}
