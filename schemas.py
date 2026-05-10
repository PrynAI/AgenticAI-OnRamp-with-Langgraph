"""Pydantic schemas for the agent's structured LLM outputs.

LangChain exposes these models to the chat model as tool schemas. The model does
not execute Python code here; it fills these fields as tool-call arguments. The
rest of the graph can then pass the model's structured fields to Tavily and to
the final answer extractor without fragile string parsing.
"""

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    """Model self-critique used to guide the search and revision step."""

    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")


class AnswerQuestion(BaseModel):
    """Structured output expected from the first LLM response."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    # Keep the search fan-out small so each graph iteration stays bounded and
    # Tavily results remain focused on the critique.
    search_queries: list[str] = Field(
        min_length=1,
        max_length=3,
        description=(
            "Search queries for researching improvements that address the "
            "critique of the current answer."
        ),
    )


class ReviseAnswer(AnswerQuestion):
    """Structured output expected from the citation-backed revision."""

    # This extends the initial answer schema because the revision still needs an
    # answer, reflection, and optional follow-up searches, plus citations.
    references: list[str] = Field(
        min_length=1,
        description="Citations motivating your updated answer.",
    )
