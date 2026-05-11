from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from typing import Literal

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import (
    GENERATE,
    GRADE_DOCUMENTS,
    MAX_GENERATION_ATTEMPTS,
    MAX_WEB_SEARCH_ATTEMPTS,
    RETRIEVE,
    WEBSEARCH,
)
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import AgenticRagState

load_dotenv()

DocumentDecision = Literal["generate", "websearch"]
GenerationDecision = Literal["not supported", "useful", "not useful", "max retries"]
RouteDecision = Literal["retrieve", "websearch"]


def decide_to_generate(state: AgenticRagState) -> DocumentDecision:
    print("---ASSESS GRADED DOCUMENTS---")

    if state.get("web_search", False):
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH

    print("---DECISION: GENERATE---")
    return GENERATE


def grade_generation_grounded_in_documents_and_question(
    state: AgenticRagState,
) -> GenerationDecision:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    generation_attempts = state.get("generation_attempts", 0)
    web_search_attempts = state.get("web_search_attempts", 0)

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"

        if web_search_attempts >= MAX_WEB_SEARCH_ATTEMPTS:
            print("---DECISION: MAX WEB SEARCH ATTEMPTS REACHED---")
            return "max retries"
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"

    if generation_attempts >= MAX_GENERATION_ATTEMPTS:
        print("---DECISION: MAX GENERATION ATTEMPTS REACHED---")
        return "max retries"
    print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
    return "not supported"


def route_question(state: AgenticRagState) -> RouteDecision:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

    raise ValueError(f"Unsupported datasource: {source.datasource}")


workflow = StateGraph(AgenticRagState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
        "max retries": END,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="images\\AdvancedAgenticRAG.png")
