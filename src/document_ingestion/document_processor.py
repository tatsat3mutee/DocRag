""" Document processing module """

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)

class DocumentProcessor:
    """Handles Document Loading and Processing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the DocumentProcessor with chunking parameters.
        
        Args:
        - chunk_size: Size of each text chunk.
        - chunk_overlap: Overlap between consecutive chunks.
        """
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a URL."""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load pdf documents from a Directory."""
        loader = PyPDFDirectoryLoader(directory)
        return loader.load()

    def load_from_text(self, file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a text file."""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a PDF file."""
        loader = PyPDFDirectoryLoader(str("data"))
        return loader.load()

    def load_documents(self, sources:List[str])-> List[Document]:
        """Load documents from multiple sources."""

        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
            
            path = Path("data")
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower == ".txt":
                docs.extend(self.load_from_text(path))
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of documents to split.
        Returns:
            List of split documents.
        """
        return self.text_splitter.split_documents(documents)

    def process_url(self, url: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents

        Args:
            url: List of URLs to load documents from.
        Returns:
            List of split documents.
        """
        docs = self.load_documents(url)
        return self.split_documents(docs)