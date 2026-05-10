"""Define the LLM prompts and structured chains used by the graph.

This module owns the model-facing part of the workflow. It creates a shared
research prompt template, then specializes it into two chains:
`first_responder` for the initial answer and `revisor` for the citation-backed
revision after search results are available.

Both chains force the model to respond through Pydantic-backed tool calls. That
keeps the graph output predictable: downstream nodes can read `answer`,
`reflection`, `search_queries`, and `references` from structured arguments
instead of parsing free-form text.
"""

import datetime

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer

# Load API keys before constructing LangChain clients that read from the env.
load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-nano")

# Used by the local smoke test in the __main__ block. The graph itself relies
# on tool calls directly rather than parsing this chain output.
pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

# The actor prompt is shared by both LLM calls. `first_instruction` is the
# extension point that changes the model's job from first draft to revision.
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="""You are an expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer."""),
        MessagesPlaceholder(variable_name="messages"),
        # This final instruction helps keep the model aligned with the schema
        # selected in `bind_tools`.
        SystemMessage(
            content="Answer the user's question above using the required format."
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)


# Tool choice is forced so downstream ToolNode always receives structured args.
# The generated `search_queries` field drives the next graph node.
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """Revise your previous answer using the new information.
- Use the previous critique to add important missing information.
- Include numerical citations in the revised answer so it can be verified.
- Add a "References" section at the bottom of the answer. It does not count toward the word limit.
- Format references as:
  - [1] https://example.com
  - [2] https://example.com
- Remove superfluous information and keep the revised answer under 250 words."""


# The revisor uses the same conversation state as the first responder, plus the
# ToolMessage containing Tavily results. Its schema adds `references`.
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC/autonomous soc problem domain,"
        "list startups that do that and raised capital."
    )

    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | pydantic_parser
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
