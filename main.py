from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from chains import generate_chain, reflect_chain


class MessageGraph(TypedDict):
    # add_messages appends each node response to the existing conversation state.
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    # The generator sees the full message history, including previous critiques.
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    # Treat critique as human feedback so the generator revises against it next.
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    # Stop after several generate/reflect cycles to avoid an unbounded loop.
    if len(state["messages"]) > 6:
        return END

    return REFLECT


# After each generation, either end or reflect. Reflection always loops back.
builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

# Print visualizations so the workflow shape is easy to inspect while learning.
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()


if __name__ == "__main__":
    print("Hello Langgraph")
    # Replace this message to test the reflection loop with different copy.
    inputs = {"messages": [HumanMessage(content="""Make this Linkedin better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post""")]}
    response = graph.invoke(inputs)
    print(response["messages"][-1].content)
