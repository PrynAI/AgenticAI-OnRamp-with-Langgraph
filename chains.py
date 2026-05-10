"""LangChain prompt chains for LinkedIn post generation and critique.

The module defines a generator chain for drafting posts and a reflector chain
for reviewing those drafts. Each chain can use a different chat model so the
caller can balance generation cost against critique quality in a reflection
loop.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

# Load API keys and provider configuration before the chat models are
# initialized.
load_dotenv()

# Generator uses the smaller model for lower-cost draft creation.
llm_generator = ChatOpenAI(model="gpt-5.4-nano", temperature=0)

# Reflector uses a stronger model with higher reasoning effort because critique
# quality has an outsized impact on the next generated draft.
llm_reflector = ChatOpenAI(model="gpt-5.4-mini", temperature=0, reasoning_effort="high")

# Generator prompt: produces or revises the LinkedIn post based on the message
# history passed by the caller.
generator_prompt_template = ChatPromptTemplate(
    [
        SystemMessage(
            content="You are a twitter techie influencer assistant tasked with writing excellent linkedin  posts."
            " Generate the best linkedin post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Reflector prompt: critiques the generated post and returns actionable
# recommendations that can be fed back into the generator.
reflector_prompt_template = ChatPromptTemplate(
    [
        SystemMessage(
            content="You are a viral linkdedin influencer grading a linkedin posts. Generate critique and recommendations for the user's linkdedin post."
            "Always provide detailed recommendations, including requests for length, virality, style, etc"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Expose runnable chains that pair each prompt template with its task-specific
# model.
generator_chain = generator_prompt_template | llm_generator
reflector_chain = reflector_prompt_template | llm_reflector
