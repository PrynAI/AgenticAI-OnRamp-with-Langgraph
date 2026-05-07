from typing import Any, Dict
from retriever import retriever
from graph.state import GraphState


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("-----Retrieving-------")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
