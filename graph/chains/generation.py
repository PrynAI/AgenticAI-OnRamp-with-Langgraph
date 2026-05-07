from dotenv import load_dotenv
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

client = Client()

try:
    prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)
except TypeError:
    prompt = client.pull_prompt("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
