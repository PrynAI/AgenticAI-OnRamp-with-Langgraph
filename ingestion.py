from dotenv import load_dotenv

load_dotenv()
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
import weaviate
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from weaviate.classes.init import Auth

WEAVIATE_COLLECTION_NAME = os.getenv(
    "WEAVIATE_COLLECTION_NAME", "Langgraphwebcollection"
)

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
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

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
