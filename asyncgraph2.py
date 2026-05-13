from dotenv import load_dotenv

load_dotenv()
import time

import operator
from typing import Annotated, Any, Sequence

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class AsyncGraphState(TypedDict):
    # Every node returns a small state update like {"aggregate": ["I'm A"]}.
    # `operator.add` tells LangGraph how to merge list updates from multiple
    # branches into one combined list instead of overwriting the field.
    aggregate: Annotated[list, operator.add]

    # This input field decides which pair of branch nodes runs after node "a".
    which: str


class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: AsyncGraphState) -> Any:
        # Sleeping makes the node execution order easier to observe in the console.
        time.sleep(1)

        # The node receives the current graph state, prints what it is adding,
        # then returns only the partial state update it wants to contribute.
        print(f"Adding {self._value} to {state['aggregate']})")
        return {"aggregate": [self._value]}


# Build the graph definition before compiling it into an executable graph.
builder = StateGraph(AsyncGraphState)

# Node "a" is always the first real node after START.
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_edge(START, "a")

# These nodes are potential downstream steps. Only some of them run per request.
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))
builder.add_node("e", ReturnNodeValue("I'm E"))


def route_bc_or_cd(state: AsyncGraphState) -> Sequence[str]:
    # After node "a" finishes, inspect the input state and choose the next
    # branch set. Returning two node names fans execution out to both nodes.
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]


intermediates = ["b", "c", "d"]

# Conditional routing happens immediately after node "a":
# - which == "cd" -> run "c" and "d"
# - anything else -> run "b" and "c"
# `intermediates` declares the possible destinations for this routing step.
builder.add_conditional_edges(
    "a",
    route_bc_or_cd,
    intermediates,
)

# Whichever branch nodes were selected all feed into the same downstream node "e".
for node in intermediates:
    builder.add_edge(node, "e")

# Node "e" is the final processing step before the graph terminates.
builder.add_edge("e", END)

# Turn the graph definition into an executable graph object and also render
# a diagram so the branch structure can be inspected visually.
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="images\\asyncgraph2.png")


if __name__ == "__main__":
    print("Hello Async Graph")

    # This sample input starts with an empty aggregate and an empty `which`
    # value, so routing follows the default ["b", "c"] branch.
    #
    # Expected high-level flow:
    # 1. "a" adds "I'm A"
    # 2. "b" and "c" run from the conditional branch
    # 3. "e" runs after those selected branches finish
    # 4. END stops the graph
    graph.invoke({"aggregate": [], "which": ""}, {"configurable": {"thread_id": "foo"}})
