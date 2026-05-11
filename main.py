from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    # Import after loading .env because graph construction initializes clients.
    from graph.graph import app

    print("Hello Advanced RAG")
    print(app.invoke({"question": "agent memory?"}))


if __name__ == "__main__":
    main()
