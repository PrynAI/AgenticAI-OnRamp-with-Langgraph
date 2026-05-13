from dotenv import load_dotenv

load_dotenv()
from graph import app

if __name__ == "__main__":
    # Thread IDs identify independent saved runs inside the SQLite checkpointer.
    thread = {"configurable": {"thread_id": "1"}}

    # This is the initial state that enters the graph at START.
    initial_input = {"input": "Hello World"}

    # Start the graph. Because `graph.py` configured an interrupt before
    # `human_feedback`, streaming stops after `step_1` and leaves a checkpoint.
    for event in app.stream(initial_input, thread, stream_mode="values"):
        print(event)

    # Show which node is scheduled to run next once the graph resumes.
    print(app.get_state(thread).next)

    user_input = input("Tell me how you want to update the state: ")

    # Attach the human response to the paused thread as if the
    # `human_feedback` node had produced this state update.
    app.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")

    print("--State after update--")
    print(app.get_state(thread))

    # Confirm the graph now points at the next executable node.
    print(app.get_state(thread).next)

    # Resume from the saved checkpoint. Passing `None` means "continue from the
    # existing checkpointed state" instead of starting a brand-new run.
    for event in app.stream(None, thread, stream_mode="values"):
        print(event)
