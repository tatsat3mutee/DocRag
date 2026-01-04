""" RAG State Module """

from typing import TypedDict, List
from langchain_core.documents import Document

class RAGState(TypedDict):
    """State for RAG application"""

    question: str
    retrieve_docs: List[Document]
    answer: str