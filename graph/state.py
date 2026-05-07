from typing import TypedDict, List, NotRequired
from langchain_core.documents import Document


class GraphState(TypedDict):
    """State that contains attriutes with defined types that flows through out state of the graph

    Attributes:
        question:str:question:input prompt to user
        generation:str:LLM generation : response to the user
        web_search:bool:dection to call websearch or not
        document:list of document that generate from retrieval & websearch


    """

    question: str
    generation: NotRequired[str]
    web_search: NotRequired[bool]
    documents: NotRequired[List[Document]]
    generation_attempts: NotRequired[int]
    web_search_attempts: NotRequired[int]
    web_search_query: NotRequired[str]
