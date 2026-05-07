from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("-----GENERATE......")
    question = state["question"]
    documents = state.get("documents", [])
    generation_attempts = state.get("generation_attempts", 0) + 1

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "generation_attempts": generation_attempts,
    }
