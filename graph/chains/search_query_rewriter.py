from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

system = """You rewrite failed RAG questions into concise web search queries.
Use the original user question and the previous answer to identify what information is missing.
Return only the search query, with no explanation."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Original question:\n{question}\n\nPrevious answer that was graded not useful:\n{generation}",
        ),
    ]
)

search_query_rewriter = prompt | llm | StrOutputParser()
