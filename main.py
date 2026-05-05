from dotenv import load_dotenv
import os

load_dotenv()


def main():
    print("Hello from agenticai-onramp-with-langgraph!")
    print(f"API Key loaded: {os.getenv('OPENAI_API_KEY')}")


if __name__ == "__main__":
    main()
