import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline_ollama import RAGPipeline

# Load environment variables FIRST before anything else
load_dotenv(override=True)

st.set_page_config(page_title="RAG Chatbot (Ollama)", layout="wide")
st.title("🤖 RAG Chatbot - Ollama Edition")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    try:
        with st.spinner("Initializing RAG Pipeline with Ollama..."):
            st.session_state.rag_pipeline = RAGPipeline()
            model_name = os.getenv("MODEL", "mistral")
            st.success(f"✅ RAG Pipeline initialized with model: **{model_name}**")
    except Exception as e:
        st.error(f"❌ Failed to initialize pipeline: {str(e)}")
        st.session_state.rag_pipeline = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me a question..."):
    if st.session_state.rag_pipeline is None:
        st.error("❌ RAG Pipeline not initialized. Please refresh the page.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_pipeline.query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Make sure Ollama is running: `ollama serve`")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    model_name = os.getenv("MODEL", "mistral")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    st.markdown(f"""
    This is a **RAG Chatbot** powered by **Ollama**.
    
    **Current Model:** `{model_name}`
    **Ollama URL:** `{ollama_url}`
    
    ### Features:
    - 📄 Document retrieval & augmented generation
    - 🔍 Vector-based semantic search
    - 💬 Context-aware responses
    """)
