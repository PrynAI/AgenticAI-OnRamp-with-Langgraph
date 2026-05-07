from typing import TypedDict, List


class GraphState(TypedDict):
    """State that contains attriutes with defined types that flows through out state of the graph

    Attributes:
        question:str:question:input prompt to user
        generation:str:LLM generation : response to the user
        web_search:bool:dection to call websearch or not
        document:list of document that generate from retrieval & websearch


    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
