from dotenv import load_dotenv

load_dotenv()

import operator
from typing import Annotated, Any
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import time


class AsynGraphState(TypedDict):
    # Each node contributes one list item, and LangGraph merges all of those
    # list updates into one shared aggregate state value.
    aggregate: Annotated[list, operator.add]


class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: AsynGraphState) -> Any:
        # The delay makes branch execution easier to observe in terminal output.
        time.sleep(1)

        # Nodes receive the current shared state, then return only the partial
        # update they want LangGraph to merge back into that state.
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}


# Build a graph that demonstrates fan-out, a downstream step, and a join.
builder = StateGraph(AsynGraphState)

# Node "a" always starts the workflow after START.
builder.add_node("a", ReturnNodeValue("I'm A"))
builder.add_edge(START, "a")

# These nodes make up the two branches that start after "a".
builder.add_node("b", ReturnNodeValue("I'm B"))
builder.add_node("b2", ReturnNodeValue("I'm B2"))
builder.add_node("c", ReturnNodeValue("I'm C"))
builder.add_node("d", ReturnNodeValue("I'm D"))

# After "a", execution fans out:
# - branch 1 runs "b" and then "b2"
# - branch 2 runs "c"
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "b2")

# Node "d" waits on both "b2" and "c", so it models a join after parallel work.
builder.add_edge(["b2", "c"], "d")

# "d" is the final processing step before the graph ends.
builder.add_edge("d", END)

# Compile the graph and render a diagram of the final topology.
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="images\\asyncgraph1.png")


if __name__ == "__main__":
    print("Hello Async Graph")

    # High-level flow for this invocation:
    # START -> a -> (b -> b2) and c -> d -> END
    # The aggregate list keeps the values returned by each executed node.
    graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "foo"}})
