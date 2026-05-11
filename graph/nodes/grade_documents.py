from typing import Any

from langchain_core.documents import Document

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import AgenticRagState


def grade_documents(state: AgenticRagState) -> dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state

    """
    print("-----CHECK DOCUMENT RELEVANCE TO QUESTION....")

    question = state["question"]
    documents = state.get("documents", [])

    filtered_docs: list[Document] = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )

        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE:DOCUMENT RELEVANT----")
            filtered_docs.append(d)
        else:
            print("----GRADE:DOCUMENT NOT RELEVANT...")
            # One weak document is enough to repair context with web search.
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
