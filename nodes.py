from state import PersistantMemoryState


def step_1(state: PersistantMemoryState) -> None:
    """First automated step in the graph."""
    # This example node only prints. Returning `None` means it does not mutate
    # the graph state before execution pauses for human input.
    print("-----Step 1------")


def human_feedback(state: PersistantMemoryState) -> None:
    """Placeholder node for human-provided input."""
    # `main.py` injects the actual `user_feedback` state update while execution
    # is paused before this node, then resumes the workflow.
    print("-----Human Feedback------")


def step_3(state: PersistantMemoryState) -> None:
    """Final automated step after human feedback state is added."""
    # This is the final visible step before the graph reaches END.
    print("-------Step3-------")
