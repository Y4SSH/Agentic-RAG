from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    generation: str
    documents: list[Document]
    search_loop_count: int
    steps_taken: list[str]
