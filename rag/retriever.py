from langchain_core.vectorstores import VectorStoreRetriever

from rag.ingestion import vectorstore

retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": 3})
