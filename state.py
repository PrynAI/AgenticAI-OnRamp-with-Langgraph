from typing import TypedDict


class PersistantMemoryState(TypedDict):
    """state graph values that graph carries between nodes"""

    input: str
    user_Feedback: str
