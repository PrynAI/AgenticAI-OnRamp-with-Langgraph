from dotenv import load_dotenv

load_dotenv()
from langgraph.graph import StateGraph, START, END
from state import PersistantMemoryState
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from nodes import step_1, step_3, human_feedback

builder = StateGraph(PersistantMemoryState)

builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

conn = sqlite3.connect("LTM_PersistanceDB.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
app = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])
app.get_graph().draw_mermaid_png(output_file_path="images\\graph.png")
