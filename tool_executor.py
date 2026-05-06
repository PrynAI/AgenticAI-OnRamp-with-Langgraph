"""LangGraph tool node for running the model-generated search queries."""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, RevisedAnswer

tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run each generated query through Tavily and return batched results.

    The LLM tool calls also contain fields such as ``answer`` and
    ``reflection``. ``**kwargs`` intentionally absorbs those fields because
    this node only needs the generated ``search_queries`` list.
    """
    return tavily_tool.batch([{"query": query} for query in search_queries])


# Tool names must match the schema/tool-call names produced by the chains.
# ToolNode uses those names to route AnswerQuestion and RevisedAnswer calls here.
execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=RevisedAnswer.__name__),
    ]
)
