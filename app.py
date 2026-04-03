"""
Streamlit UI for LexiFlow - Intelligent AI RAG Engine
"""

import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
import tempfile
from pathlib import Path

# Page config
st.set_page_config(
    page_title="LexiFlow - AI RAG Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        font-weight: bold;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .source-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        border-left: 4px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("# 🤖 LexiFlow - Intelligent AI RAG Engine")
st.markdown("### Powered by Advanced RAG (Retrieval-Augmented Generation)")

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
    st.session_state.chat_history = []
    st.session_state.documents_loaded = False

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Model selection
    st.markdown("### 🧠 Model Selection")
    use_local = st.checkbox("🏠 Use Local Ollama (Free)", value=False, help="Use free local LLM instead of OpenAI API", key="use_local_checkbox")
    
    if use_local:
        st.info("💡 **Memory Note:** For best performance with limited RAM (<2GB), use 'neural-chat' or 'dolphin-mixtral'. First run will download the model.")
        model = st.selectbox(
            "Local Model (Select by available RAM)",
            ["qwen2:0.5b", "phi", "neural-chat", "dolphin-mixtral", "mistral", "llama2"],
            help="qwen2:0.5b: <1GB | phi: 2GB | neural-chat: 4GB | mistral: 4.5GB | llama2: 7GB",
            key="local_model_select"
        )
        api_key = "local"  # Dummy value for local mode
    else:
        model = st.selectbox(
            "🧠 LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            help="Select the language model for answer generation",
            key="openai_model_select"
        )
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Get your API key from https://platform.openai.com/api-keys",
            key="api_key_input"
        )
    
    st.markdown("### 📊 Chunk Settings")
    chunk_size = st.slider("Chunk Size (tokens)", 100, 1000, 500, step=50, key="chunk_size_slider")
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, step=10, key="chunk_overlap_slider")
    
    # Initialize RAG if API key provided or local mode enabled
    if (api_key and api_key != "local") or use_local:
        if st.session_state.rag_pipeline is None:
            try:
                st.session_state.rag_pipeline = RAGPipeline(api_key=api_key, model=model, use_local=use_local)
                st.session_state.rag_pipeline.chunk_size = chunk_size
                st.session_state.rag_pipeline.chunk_overlap = chunk_overlap
                if use_local:
                    st.success("✅ Local Ollama Pipeline Initialized!")
                else:
                    st.success("✅ RAG Pipeline Initialized!")
            except Exception as e:
                st.error(f"❌ Pipeline Error: {str(e)}")
    
    st.markdown("---")
    
    # Clear all data button
    if st.button("🗑️ Clear All Data", key="clear_btn"):
        if st.session_state.rag_pipeline:
            st.session_state.rag_pipeline.clear()
        st.session_state.chat_history = []
        st.session_state.documents_loaded = False
        st.success("✅ All data cleared!")
        st.rerun()

# Main content area
if not use_local and not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar OR enable 'Use Local Ollama' for free!")
    st.info("📌 Options:\n1. Get OpenAI API key at https://platform.openai.com/api-keys\n2. Or use free local Ollama (check sidebar)")
elif use_local:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Learn", "💬 Chat", "📚 Advanced"])
    
    # TAB 1: Upload & Learn
    with tab1:
        st.markdown("## 📤 Load Your Knowledge Base")
        
        # PDF Upload
        st.markdown("### Upload PDF Files")
        pdf_files = st.file_uploader(
            "Choose PDF file(s)",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDF documents that will be converted to knowledge"
        )
        
        if pdf_files and st.button("📥 Process PDF(s)", key="pdf_btn"):
            with st.spinner("Processing PDFs..."):
                for pdf_file in pdf_files:
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf_file.read())
                        tmp_path = tmp.name
                    
                    # Load PDF
                    docs = st.session_state.rag_pipeline.load_pdf(tmp_path)
                    if docs:
                        if st.session_state.documents_loaded:
                            st.session_state.rag_pipeline.add_documents(docs)
                        else:
                            st.session_state.rag_pipeline.create_vector_db(docs)
                            st.session_state.documents_loaded = True
                    
                    # Clean up
                    os.remove(tmp_path)
            
            st.success(f"✅ Loaded {len(pdf_files)} PDF(s)! Ready to ask questions.")
        
        st.markdown("---")
        
        # Website URL
        st.markdown("### 🌐 Load from Website")
        url = st.text_input("Enter website URL", placeholder="https://example.com")
        
        if url and st.button("🔗 Load Website", key="url_btn"):
            with st.spinner("Fetching and processing website..."):
                docs = st.session_state.rag_pipeline.load_website(url)
                if docs:
                    if st.session_state.documents_loaded:
                        st.session_state.rag_pipeline.add_documents(docs)
                    else:
                        st.session_state.rag_pipeline.create_vector_db(docs)
                        st.session_state.documents_loaded = True
            
            st.success("✅ Website loaded! Ready to ask questions.")
        
        # Status
        if st.session_state.documents_loaded:
            st.markdown("---")
            st.success("✅ Knowledge base loaded successfully!")
            st.info(f"📊 Total documents: {len(st.session_state.rag_pipeline.documents)}")
    
    # TAB 2: Chat
    with tab2:
        st.markdown("## 💬 Ask Your AI Assistant")
        
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please load documents first using the 'Upload & Learn' tab.")
        else:
            # Display chat history
            st.markdown("### 📋 Conversation History")
            
            # Chat container
            chat_container = st.container()
            
            with chat_container:
                for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q{i+1}: {question[:50]}...", expanded=(i == len(st.session_state.chat_history)-1)):
                        st.markdown(f"**Question:** {question}")
                        st.markdown(f"""
                        <div class="answer-box">
                            <b>Answer:</b><br>
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show sources
                        with st.expander("📎 Show Sources"):
                            for j, source in enumerate(sources):
                                st.markdown(f"""
                                <div class="source-box">
                                    <b>Source {j+1}:</b><br>
                                    {source.page_content[:300]}...
                                </div>
                                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Query input
            st.markdown("### ❓ Ask a Question")
            question = st.text_area(
                "Type your question here",
                placeholder="What is this document about?",
                height=100,
                key="query_input"
            )
            
            # Number of sources
            top_k = st.slider("Number of source documents to retrieve", 1, 10, 3)
            
            # Submit button
            col1, col2 = st.columns([3, 1])
            with col1:
                submit_btn = st.button("🚀 Ask Question", use_container_width=True)
            
            if submit_btn and question:
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.rag_pipeline.query(question, top_k=top_k)
                    
                    # Store in history
                    st.session_state.chat_history.append(
                        (question, response["answer"], response["sources"])
                    )
                    
                    # Display result
                    st.success("✅ Answer generated!")
                    
                    st.markdown(f"""
                    <div class="answer-box">
                        <b>🤖 AI Answer:</b><br>
                        {response['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources
                    st.markdown("### 📎 Source Documents")
                    for i, source in enumerate(response["sources"]):
                        with st.expander(f"Source {i+1}"):
                            st.write(source.page_content)
                    
                    st.rerun()
    
    # TAB 3: Advanced Features
    with tab3:
        st.markdown("## 🚀 Advanced Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💾 Save/Load Vector DB")
            
            db_path = st.text_input("Database path", value="./vector_db")
            
            col_save, col_load = st.columns(2)
            with col_save:
                if st.button("💾 Save DB", use_container_width=True):
                    if st.session_state.documents_loaded:
                        st.session_state.rag_pipeline.save_db(db_path)
                        st.success("✅ Database saved!")
                    else:
                        st.warning("⚠️ No data to save")
            
            with col_load:
                if st.button("📂 Load DB", use_container_width=True):
                    st.session_state.rag_pipeline.load_db(db_path)
                    st.session_state.documents_loaded = True
                    st.success("✅ Database loaded!")
        
        with col2:
            st.markdown("### 🔍 Similarity Search")
            
            search_query = st.text_input("Search the knowledge base")
            num_results = st.slider("Number of results", 1, 10, 5)
            
            if search_query and st.button("🔎 Search", use_container_width=True):
                if st.session_state.documents_loaded:
                    results = st.session_state.rag_pipeline.similarity_search(search_query, k=num_results)
                    
                    st.markdown(f"### 📊 Found {len(results)} results")
                    for i, (doc, score) in enumerate(results):
                        with st.expander(f"Result {i+1} (Score: {score:.2f})"):
                            st.write(doc.page_content)
                else:
                    st.warning("⚠️ Load documents first")
        
        st.markdown("---")
        
        # Chat memory and context
        st.markdown("### 💭 Conversation Insights")
        
        if st.session_state.chat_history:
            st.info(f"📈 Total questions asked: {len(st.session_state.chat_history)}")
            
            # Show longest question
            longest_q = max(st.session_state.chat_history, key=lambda x: len(x[0]))[0]
            st.write(f"**Longest question:** {longest_q[:100]}...")
        else:
            st.info("💬 No conversation history yet")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 11px; padding: 20px;">
    <p>🚀 LexiFlow | Powered by LangChain, OpenAI & FAISS</p>
    <p>Built for AI Excellence | Version 1.0</p>
</div>
""", unsafe_allow_html=True)
