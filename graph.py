from dotenv import load_dotenv

load_dotenv()
from langgraph.graph import StateGraph, START, END
from state import PersistantMemoryState
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from nodes import step_1, step_3, human_feedback

# Define the graph structure separately from the script that drives it. This
# makes `main.py` responsible for execution while this file owns topology.
builder = StateGraph(PersistantMemoryState)

# Register the three logical steps in the workflow.
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)

# The workflow is linear:
# START -> step_1 -> human_feedback -> step_3 -> END
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

# Store graph checkpoints in a local SQLite database so a paused execution can
# later be inspected, updated, and resumed.
conn = sqlite3.connect("LTM_PersistanceDB.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

# Pause execution immediately before the `human_feedback` node. The graph state
# remains checkpointed so `main.py` can inject the human response and resume.
app = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

# Render the graph topology for visual inspection.
app.get_graph().draw_mermaid_png(output_file_path="images\\graph.png")
