from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    empty_response,
    fallback_notice,
    generate,
    grade_documents,
    hallucination_check,
    retrieve,
    rewrite_query,
)
from src.agent.state import GraphState
from src.config import get_settings

RouteLabel = Literal[
    "rewrite_query", "generate", "empty_response", "fallback_notice", END
]


def _hallucination_failure_count(steps_taken: list[str]) -> int:
    return sum(
        1
        for step in steps_taken
        if step.lower().startswith("hallucination check failed")
    )


def _route_after_grading(state: GraphState) -> str:
    settings = get_settings()
    if int(state.get("search_loop_count", 0)) > settings.search_loop_limit:
        return "empty_response" if not state.get("documents") else "fallback_notice"
    if not state.get("documents"):
        return "empty_response"
    return "generate"


def _route_after_hallucination(state: GraphState) -> str:
    settings = get_settings()
    steps_taken = list(state.get("steps_taken", []))
    if steps_taken and steps_taken[-1].lower().startswith("hallucination check passed"):
        return END
    if (
        steps_taken
        and "grounding audit unavailable: model unavailable" in steps_taken[-1].lower()
    ):
        return "fallback_notice"
    if int(state.get("search_loop_count", 0)) > settings.search_loop_limit:
        return "fallback_notice"
    if _hallucination_failure_count(steps_taken) >= 2:
        return "fallback_notice"
    return "generate"


@lru_cache(maxsize=1)
def build_graph():
    workflow: StateGraph[GraphState] = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("empty_response", empty_response)
    workflow.add_node("generate", generate)
    workflow.add_node("hallucination_check", hallucination_check)
    workflow.add_node("fallback_notice", fallback_notice)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("empty_response", END)
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_edge("fallback_notice", END)

    workflow.add_conditional_edges(
        "grade_documents",
        _route_after_grading,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
            "empty_response": "empty_response",
            "fallback_notice": "fallback_notice",
        },
    )
    workflow.add_conditional_edges(
        "hallucination_check",
        _route_after_hallucination,
        {
            "generate": "generate",
            "fallback_notice": "fallback_notice",
            END: END,
        },
    )
    return workflow.compile()


@lru_cache(maxsize=1)
def _compiled_graph():
    return build_graph()


def run_agent(question: str) -> GraphState:
    initial_state: GraphState = {
        "question": question,
        "generation": "",
        "documents": [],
        "search_loop_count": 0,
        "steps_taken": [f"Initialized agent for question: {question}"],
    }
    return _compiled_graph().invoke(initial_state)
