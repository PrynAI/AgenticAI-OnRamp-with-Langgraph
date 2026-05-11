from typing import Any

from rag.retriever import retriever
from graph.state import AgenticRagState


def retrieve(state: AgenticRagState) -> dict[str, Any]:
    print("-----Retrieving-------")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
