from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from llm import tools, llm

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
Use the Tavily search tool to search for the latest information from the internet.
Use sources like AccuWeather or Weather.com when fetching live weather information.
Retrieve the current temperature in Celsius.
Use the triple tool when a calculation requires multiplying a numeric result by three.

"""


def run_agent_reasoning(state: MessagesState) -> dict:
    """Call the tool-bound LLM and append its response to the message state."""
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}


# ToolNode reads tool calls from the latest AI message, executes the matching
# tools, and appends the tool results back into the graph state.
tool_node = ToolNode(tools)
