"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # Model Configuration
    # Use tool-use optimized model for agent workflows
    LLM_MODEL = "qwen/qwen3-32b"
    EMBEDDING_MODEL = os.getenv("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model
        
        Returns:
            ChatGroq: Initialized Groq chat model instance
            
        Raises:
            ValueError: If GROQ_API_KEY is not set
            Exception: If model initialization fails
        """
        # Validate API key exists
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please ensure it's defined in your .env file"
            )
        
        try:
            # Set environment variable (ChatGroq will read from it)
            os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
            
            # Initialize and return the ChatGroq model
            return ChatGroq(
                model=cls.LLM_MODEL
            )
        except Exception as e:
            raise Exception(
                f"Failed to initialize LLM model '{cls.LLM_MODEL}': {str(e)}"
            ) from e
    
    @classmethod
    def get_embedding_config(cls) -> dict:
        """Get embedding configuration
        
        Returns:
            dict: Configuration for HuggingFace embeddings
        """
        from typing import Any, Dict
        
        config: Dict[str, Any] = {
            "model_name": cls.EMBEDDING_MODEL
        }
        
        # Add API token if available
        if cls.HUGGINGFACE_API_KEY:
            config["model_kwargs"] = {"token": cls.HUGGINGFACE_API_KEY}
        
        return config
