from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import tools, llm

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assitant that can use tools to answer questions.
Use 'TavilySearch' tool to search latest information from internet.
User sources like (e.g., AccuWeather, Weather.com) to fect live information
Use 'triple' tool calcuate result from tavily search tool

"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}


tool_node = ToolNode(tools)
