from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Generator prompt: produces the first draft and later revisions.
generation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a Linkedin techie influencer assistant tasked with writing excellent linkedin posts."
            " Generate the best linkedin post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts."
        ),
        # LangGraph passes the accumulated conversation into this placeholder.
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Reflection prompt: critiques the latest generated post for the next iteration.
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a viral Linkedin influencer grading a tweet. Generate critique and recommendations for the user's post."
            "Always provide detailed recommendations, including requests for length, virality, style, etc."
        ),
        # The reflector needs the same history to evaluate the current draft in context.
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Reuse the same chat model instance for both prompt pipelines.
llm = ChatOpenAI()

generate_chain = generation_prompt | llm

reflect_chain = reflection_prompt | llm
