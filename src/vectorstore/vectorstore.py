""" Vector Store Module """

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from src.config.config import Config

class VectorStore:
    """Manages Vector Store Operations"""
    def __init__(self):
        # Get embedding configuration from Config
        embedding_config = Config.get_embedding_config()
        self.embedding = HuggingFaceEmbeddings(**embedding_config)
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents

        Args:
        - documents: List of documents to add to the vector stored
        """

        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        """Get the retriever"""

        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4)-> List[Document]:
        """Retrieve documents based on query

        Args:
        - query: Query to search for
        - k: Number of documents to retrieve
        Returns: List of retrieved documents
        """

        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)