"""Streamlit UI for Agentic RAG System - Simplified Version"""

import streamlit as st
from pathlib import Path
import sys
import time
import tempfile
import io

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
from langchain_core.documents import Document

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        
        # Use default URLs
        urls = Config.DEFAULT_URLS
        
        # Process documents
        documents = doc_processor.process_url(urls)
        
        # Create vector store
        vector_store.create_vectorstore(documents)
        
        # Store in session for later updates
        st.session_state.vector_store = vector_store
        st.session_state.doc_processor = doc_processor
        st.session_state.llm = llm
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def add_documents_to_store(uploaded_files):
    """Add uploaded documents to the vector store"""
    if not uploaded_files:
        return 0
    
    try:
        doc_processor = st.session_state.doc_processor
        vector_store = st.session_state.vector_store
        
        all_documents = []
        
        for uploaded_file in uploaded_files:
            # Handle PDF files
            if uploaded_file.name.endswith('.pdf'):
                from pypdf import PdfReader
                pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()
                
                doc = Document(
                    page_content=pdf_text,
                    metadata={"source": uploaded_file.name, "type": "pdf"}
                )
                all_documents.append(doc)
            
            # Handle text files
            elif uploaded_file.name.endswith(('.txt', '.md')):
                text_content = uploaded_file.read().decode('utf-8')
                doc = Document(
                    page_content=text_content,
                    metadata={"source": uploaded_file.name, "type": "text"}
                )
                all_documents.append(doc)
        
        if all_documents:
            # Split documents
            split_docs = doc_processor.split_documents(all_documents)
            
            # Add to existing vector store
            if vector_store.vectorstore is not None:
                vector_store.vectorstore.add_documents(split_docs)
            else:
                vector_store.create_vectorstore(split_docs)
            
            # Rebuild the graph with updated retriever
            graph_builder = GraphBuilder(
                retriever=vector_store.get_retriever(),
                llm=st.session_state.llm
            )
            graph_builder.build()
            st.session_state.rag_system = graph_builder
            
            return len(split_docs)
        
        return 0
    
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return 0

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    
    st.markdown("---")
    
    # Document Upload Section
    st.markdown("### 📤 Upload Additional Documents")
    st.markdown("Upload PDF or text files to add to your knowledge base")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        if st.button("📥 Add Documents to Knowledge Base"):
            with st.spinner("Processing uploaded documents..."):
                num_chunks = add_documents_to_store(uploaded_files)
                if num_chunks > 0:
                    st.success(f"✅ Added {num_chunks} document chunks from {len(uploaded_files)} file(s)!")
                else:
                    st.warning("No documents were processed")
    
    st.markdown("---")
    
    # Search interface
    st.markdown("### 🔎 Search")
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
    
    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()
                
                # Get answer
                result = st.session_state.rag_system.run(question)
                
                elapsed_time = time.time() - start_time
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(result['answer'])
                
                # Show retrieved docs in expander
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result['retrieve_docs'], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True
                        )
                
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
    
    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()