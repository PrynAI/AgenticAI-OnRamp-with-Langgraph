from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num: float) -> float:
    """Return three times the provided numeric value."""
    return num * 3


tools = [TavilySearch(max_results=1), triple]

# Binding tools publishes their schemas to the model so it can request them
# through OpenAI tool calling during the LangGraph loop.
llm = ChatOpenAI(model="gpt-5-nano", temperature=0).bind_tools(tools)
