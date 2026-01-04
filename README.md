# 🔍 DocRag - Agentic RAG Document Search System

A powerful Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, and Streamlit. DocRag enables intelligent document search and question-answering by combining document retrieval with large language models.

## ✨ Features

- **🌐 Multi-Source Document Loading**: Support for web URLs, PDFs, and text files
- **🤖 LangGraph Workflow**: Structured RAG pipeline using LangGraph state management
- **💾 Vector Storage**: FAISS-based vector store for efficient semantic search
- **🎯 Smart Chunking**: Recursive text splitting with configurable chunk sizes
- **🖥️ Interactive UI**: Clean Streamlit interface for document queries
- **⚡ Fast Inference**: Powered by Groq's high-performance LLM API
- **📊 Source Tracking**: View retrieved source documents for each answer

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
DocRag/
├── src/
│   ├── config/              # Configuration and API setup
│   ├── document_ingestion/  # Document loading and processing
│   ├── vectorstore/         # FAISS vector store management
│   ├── graph_builder/       # LangGraph workflow construction
│   ├── nodes/               # RAG workflow nodes
│   └── state/               # State management for LangGraph
├── data/                    # Data storage directory
├── streamlit_app.py         # Main Streamlit application
├── requirements.txt         # Project dependencies
└── pyproject.toml          # Project metadata and dependencies
```

### Workflow

1. **Document Ingestion**: Load documents from URLs, PDFs, or text files
2. **Text Chunking**: Split documents into manageable chunks with overlap
3. **Embedding**: Convert text chunks into vector embeddings using HuggingFace models
4. **Vector Storage**: Store embeddings in FAISS for efficient retrieval
5. **Query Processing**: Retrieve relevant documents based on user queries
6. **Answer Generation**: Generate contextual answers using Groq LLM

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- Groq API key ([Get it here](https://console.groq.com/keys))
- HuggingFace API token (optional, for private models)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/tatsat3mutee/DocRag.git
   cd DocRag
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Or using uv (recommended):

   ```bash
   uv pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACE_API_KEY=your_huggingface_token_here  # Optional
   HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

### Running the Application

**Launch the Streamlit app:**

```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501`

## 📖 Usage

### Using the Streamlit Interface

1. Launch the application
2. Wait for the system to initialize (loads default documents)
3. Enter your question in the search box
4. Click "🔍 Search" to get answers
5. Expand "📄 Source Documents" to view retrieved context
6. View search history at the bottom of the page

### Default Documents

The system comes pre-configured with these documents:

- Lilian Weng's blog post on LLM Agents
- Lilian Weng's blog post on Diffusion Video Models

### Customizing Document Sources

Edit `src/config/config.py` to add your own URLs:

```python
DEFAULT_URLS = [
    "https://your-url-1.com",
    "https://your-url-2.com"
]
```

## 🔧 Configuration

Key configuration options in `src/config/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `qwen/qwen3-32b` | Groq model for answer generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |

## 🛠️ Technology Stack

- **LangChain**: Framework for LLM applications
- **LangGraph**: Graph-based workflow orchestration
- **Groq**: High-performance LLM inference
- **FAISS**: Vector similarity search
- **HuggingFace**: Embedding models
- **Streamlit**: Web UI framework
- **BeautifulSoup**: Web scraping
- **PyPDF**: PDF document processing

## 📦 Project Structure

### Core Components

- **DocumentProcessor**: Handles loading and chunking of documents from various sources
- **VectorStore**: Manages FAISS vector database and document retrieval
- **GraphBuilder**: Constructs LangGraph workflow with retrieval and generation nodes
- **RAGNodes**: Implements retrieval and answer generation logic
- **Config**: Centralized configuration management

### State Management

The system uses TypedDict-based state management for LangGraph:

```python
class RAGState(TypedDict):
    question: str              # User query
    retrieve_docs: List[Document]  # Retrieved documents
    answer: str               # Generated answer
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for the powerful LLM framework
- [Groq](https://groq.com/) for fast inference
- [HuggingFace](https://huggingface.co/) for embedding models
- Lilian Weng for the excellent blog posts used as default documents

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using LangChain, LangGraph, and Streamlit**
