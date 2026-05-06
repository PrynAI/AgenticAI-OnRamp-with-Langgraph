"""Pydantic schemas used as structured tool-call outputs from the LLM."""

from typing import List

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    """Self-critique that tells the revision step what to improve or remove."""

    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Initial structured answer produced by the first responder chain."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class RevisedAnswer(AnswerQuestion):
    """Revised structured answer produced after search results are available."""

    references: List[str] = Field(
        description="Citation motivating your updated answer."
    )
