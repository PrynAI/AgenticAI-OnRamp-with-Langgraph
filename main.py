"""Build and run the LangGraph reflection/search/revision workflow.

This is the application entry point. It wires three nodes into a loop:
1. `draft` asks the LLM for an initial structured answer and search queries.
2. `execute_tools` runs those search queries through Tavily.
3. `revise` asks the LLM to improve the answer using the tool results.

The compiled `graph` can be imported by notebooks, tests, or other scripts.
Running this file directly executes `DEFAULT_QUESTION` as a smoke test.
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from chains import first_responder, revisor
from tool_executor import execute_tools

MAX_ITERATIONS = 2
DEFAULT_QUESTION = (
    "Write about AI-Powered SOC / autonomous soc problem domain, list startups "
    "that do that and raised capital."
)


def draft_node(state: MessagesState):
    """Create the first structured answer from the user's message history."""
    # MessagesState stores the full conversation under "messages". Passing the
    # whole list preserves user context for the model.
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revise_node(state: MessagesState):
    """Revise the answer after Tavily search results have been appended."""
    # At this point the state includes the first answer, its tool call, and the
    # ToolMessage returned by execute_tools. The revisor prompt can see all of it.
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Stop revising once the configured number of tool/search rounds is reached."""
    tool_rounds = sum(isinstance(item, ToolMessage) for item in state["messages"])

    # Each ToolMessage represents one completed Tavily search batch. The graph
    # always performs the first search after drafting; this condition controls
    # only additional search-and-revision rounds.
    if tool_rounds >= MAX_ITERATIONS:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)

# The graph deliberately keeps each responsibility in a separate node:
# drafting, external search, and revision. That makes the loop easy to inspect
# and keeps tool execution outside the LLM prompt code.
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)

# The first draft always happens before tool execution because the draft
# produces the search_queries that Tavily needs.
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

# After every revision, decide whether another search/revision pass is useful.
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()


def run_agent(question: str = DEFAULT_QUESTION) -> MessagesState:
    """Run the compiled graph for a single user question."""
    return graph.invoke({"messages": [HumanMessage(content=question)]})


def extract_final_answer(state: MessagesState) -> str | None:
    """Return the final structured answer from the latest AI tool call."""
    # Walk backward because the final useful answer is normally in the newest
    # ReviseAnswer tool call, not necessarily in plain text message content.
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and message.tool_calls:
            answer = message.tool_calls[0].get("args", {}).get("answer")
            if isinstance(answer, str):
                return answer
    return None


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())

    result = run_agent()
    final_answer = extract_final_answer(result)
    if final_answer:
        print(final_answer)
    print(result)
