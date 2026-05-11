from typing import NotRequired, TypedDict

from langchain_core.documents import Document


class AgenticRagState(TypedDict):
    """Shared state passed between LangGraph nodes.

    Attributes:
        question: Original user question.
        generation: LLM answer generated from the current documents.
        web_search: Whether document grading requested web search repair.
        documents: Documents accumulated from retrieval and web search.
        generation_attempts: Number of answer generation attempts.
        web_search_attempts: Number of web search attempts.
        web_search_query: Last query sent to Tavily.
    """

    question: str
    generation: NotRequired[str]
    web_search: NotRequired[bool]
    documents: NotRequired[list[Document]]
    generation_attempts: NotRequired[int]
    web_search_attempts: NotRequired[int]
    web_search_query: NotRequired[str]
