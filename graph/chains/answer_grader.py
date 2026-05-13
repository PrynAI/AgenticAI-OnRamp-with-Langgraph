from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="True if the answer addresses the question, otherwise false."
    )


llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)


system = """You are an answer grader assessing whether an answer addresses the user's question.
Return true if the answer directly responds to the question.
For short term-style questions such as "agent memory?", a definition or explanation of the term counts as addressing the question.
Return false only if the answer is off-topic or does not answer what was asked."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
