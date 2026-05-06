"""LangGraph entry point for the reflection/search/revision workflow.

The compiled graph is importable as ``graph`` for notebooks or tests. Running
this file directly executes the sample research question at the bottom.
"""

from typing import Literal
import warnings

# LangGraph 1.1.10 imports JsonPlusSerializer, which currently triggers a
# LangChainPendingDeprecationWarning from an internal Reviver() default. This
# filter is intentionally narrow so other deprecation warnings still show up.
warnings.filterwarnings(
    "ignore",
    message=(
        "The default value of `allowed_objects` will change in a future version.*"
    ),
)

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from chains import first_responder, revisor
from tool_executor import execute_tools

MAX_ITERATIONS = 2
SAMPLE_QUESTION = (
    "Write about AI-Powered SOC / autonomous soc problem domain, list startups "
    "that do that and raised capital."
)


def draft_node(state: MessagesState) -> dict:
    """Ask the model for the first answer plus critique and search queries."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessagesState) -> dict:
    """Ask the model to revise the answer using the latest tool results."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", "__end__"]:
    """Route back to search until the configured number of tool rounds is met."""
    # ToolNode appends one ToolMessage for each structured tool call it runs.
    # Counting ToolMessages is therefore a simple search-round counter here.
    tool_rounds = sum(isinstance(item, ToolMessage) for item in state["messages"])
    if tool_rounds >= MAX_ITERATIONS:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])

graph = builder.compile()


def run_agent(question: str) -> MessagesState:
    """Run the graph for one user question."""
    return graph.invoke({"messages": [{"role": "user", "content": question}]})


def print_final_answer(result: MessagesState) -> None:
    """Print the final answer stored inside the model's structured tool call."""
    last_message = result["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(last_message.tool_calls[0]["args"]["answer"])
        return

    print(last_message.content)


def main() -> None:
    """Run a sample question and print both the graph and the final result."""
    print(graph.get_graph().draw_mermaid())
    result = run_agent(SAMPLE_QUESTION)
    print_final_answer(result)
    print(result)


if __name__ == "__main__":
    main()
