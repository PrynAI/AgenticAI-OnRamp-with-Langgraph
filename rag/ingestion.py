import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
import weaviate
from weaviate.classes.init import Auth

load_dotenv()

WEAVIATE_COLLECTION_NAME = os.getenv(
    "WEAVIATE_COLLECTION_NAME", "Langgraphwebcollection"
)

SOURCE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://prynai.github.io/2025/10/16/Long-Term-Memory-LangGraph-Store.html",
]

# This module intentionally indexes at import time for the learning workflow.
docs = [WebBaseLoader(url).load() for url in SOURCE_URLS]
docs_list = [item for loaded_docs in docs for item in loaded_docs]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=20
)
doc_splits = text_splitter.split_documents(docs_list)
embeddings = OpenAIEmbeddings()

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ["WEAVIATE_URL"],
    auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
)


vectorstore = WeaviateVectorStore.from_documents(
    doc_splits,
    embeddings,
    client=weaviate_client,
    index_name=WEAVIATE_COLLECTION_NAME,
    text_key="text",
)
# weaviate_client.close()
