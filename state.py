from typing import TypedDict


class PersistantMemoryState(TypedDict):
    """State values that the persisted LangGraph workflow carries between nodes."""

    # Original user or system input that starts the workflow.
    input: str

    # Human input is added later while the graph is paused at the interrupt.
    user_feedback: str
