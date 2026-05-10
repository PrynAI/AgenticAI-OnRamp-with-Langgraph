from chains import generator_chain, reflector_chain
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph
from dotenv import load_dotenv

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"
MAX_MESSAGES = 6


# Generate the next LinkedIn post draft from the current message history.
def generation_node(state: MessagesState):
    response = generator_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


# Convert the reflector's critique into a HumanMessage so the generator treats
# it as feedback on the next pass through the loop.
def reflection_node(state: MessagesState):
    response = reflector_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}


# MessagesState stores the running conversation under state["messages"] and
# appends message updates returned by each node.
builder = StateGraph(MessagesState)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessagesState):
    # Stop once the reflection loop has enough generated drafts and critiques.
    if len(state["messages"]) > MAX_MESSAGES:
        return END
    return REFLECT


builder.add_conditional_edges(
    GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT}
)
builder.add_edge(REFLECT, GENERATE)
graph = builder.compile()


if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
    graph.get_graph().print_ascii()
    print("Hello Langgraph")
    inputs = HumanMessage(content="""Make this LinkedIn post better:"
                        #LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post""")
    response = graph.invoke({"messages": [inputs]})
    print(response["messages"][-1].content)
