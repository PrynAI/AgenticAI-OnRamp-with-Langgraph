from typing import Any, TypedDict, cast

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.chains.search_query_rewriter import search_query_rewriter
from graph.state import AgenticRagState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


class TavilyResult(TypedDict, total=False):
    content: str


class TavilyResponse(TypedDict, total=False):
    results: list[TavilyResult]


def web_search(state: AgenticRagState) -> dict[str, Any]:
    print("-----WEB SEARCH...")
    question = state["question"]
    documents = list(state.get("documents") or [])
    previous_web_search_attempts = state.get("web_search_attempts", 0)
    web_search_attempts = previous_web_search_attempts + 1

    if previous_web_search_attempts == 0:
        search_query = question
    else:
        # Retry searches are rewritten using the failed answer as feedback.
        search_query = search_query_rewriter.invoke(
            {
                "question": question,
                "generation": state.get("generation", ""),
            }
        ).strip()
        search_query = search_query or question

    print(f"-----WEB SEARCH QUERY: {search_query}")
    tavily_response = cast(
        TavilyResponse, web_search_tool.invoke({"query": search_query})
    )
    tavily_results = tavily_response.get("results", [])

    joined_tavily_result = "\n".join(
        tavily_result.get("content", "") for tavily_result in tavily_results
    )
    web_results = Document(page_content=joined_tavily_result)
    documents.append(web_results)
    return {
        "documents": documents,
        "question": question,
        "web_search_attempts": web_search_attempts,
        "web_search_query": search_query,
    }
