"""Tool execution layer for model-generated search queries.

The LLM chains do not call Tavily directly. They produce structured tool calls
whose names match the schema classes in `schemas.py`. LangGraph's `ToolNode`
uses those names to route the call here, where only `search_queries` are sent to
Tavily and the search results are appended back into the graph as ToolMessages.
"""

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

# Load Tavily credentials before the search tool is created.
load_dotenv()

# One Tavily tool instance is reused by every graph iteration. `max_results`
# controls how much evidence each generated query can return.
tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **_ignored):
    """Run the model-generated search queries through Tavily."""
    if not search_queries:
        return []

    # The LLM tool call also contains answer/reflection fields. **_ignored keeps
    # this search helper focused on the only field Tavily needs.
    return tavily_tool.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode(
    [
        # Both model schemas contain `search_queries`, so both tool-call names
        # can safely route to the same implementation.
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
