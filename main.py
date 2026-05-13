from dotenv import load_dotenv

load_dotenv()
from graph import app

if __name__ == "__main__":
    # Thread IDs identify independent conversations/runs in the checkpointer.
    thread = {"configurable": {"thread_id": "1"}}

    initial_input = {"input": "Hello World"}

    # Run until the configured interrupt point.
    for event in app.stream(initial_input, thread, stream_mode="values"):
        print(event)

    print(app.get_state(thread).next)

    user_input = input("Tell me how you want to update the state: ")

    # Attach the human response to the paused thread as if it came from the
    # human_feedback node.
    app.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")

    print("--State after update--")
    print(app.get_state(thread))

    print(app.get_state(thread).next)

    # Resume from the saved checkpoint and finish the graph.
    for event in app.stream(None, thread, stream_mode="values"):
        print(event)
