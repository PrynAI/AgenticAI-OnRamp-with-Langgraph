from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import ssl
import certifi

load_dotenv()
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question. 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """

You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system),
        HumanMessage(
            content="Retrieved document: \n\n {document} \n\n User question: {question}"
        ),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
