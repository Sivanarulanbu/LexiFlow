"""
Dynamic RAG Pipeline Script
- User inputs PDF path at runtime
- Checks if PDF exists
- Splits documents
- Creates FAISS vector DB
- Queries LLM
"""

import os
import sys
from typing import List, Dict, Any

# Fix for Windows console encoding issues with emojis
if sys.platform == "win32":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

try:
    from langchain.document_loaders import PyPDFLoader
except ImportError:
    from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.vectorstores import FAISS
except ImportError:
    from langchain_community.vectorstores import FAISS

try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain_classic.chains import RetrievalQA

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False

# Fallback embeddings for local mode
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HF_EMBEDDINGS_AVAILABLE = True
except ImportError:
    HF_EMBEDDINGS_AVAILABLE = False

# ---------------- RAG Pipeline ---------------- #
class RAGPipeline:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", use_local: bool = False):
        self.use_local = use_local
        
        if use_local:
            # Use local Ollama LLM
            if not OLLAMA_AVAILABLE:
                raise ValueError("❌ Ollama integration not available. Install: pip install ollama")
            self.llm = OllamaLLM(model=model, base_url="http://localhost:11434")
            
            # Use local embeddings
            if HF_EMBEDDINGS_AVAILABLE:
                self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            else:
                raise ValueError("❌ Install HuggingFace embeddings: pip install sentence-transformers")
        else:
            # Use OpenAI API
            if not OPENAI_AVAILABLE:
                raise ValueError("❌ OpenAI not available. Install: pip install langchain-openai")
            
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("❌ OPENAI_API_KEY not set!")

            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model=model, temperature=0.7)

        self.db = None
        self.documents = []
        self.chunk_size = 500
        self.chunk_overlap = 50

    # ---------------- Load PDF ---------------- #
    def load_pdf(self, file_path: str) -> List:
        file_path = os.path.abspath(file_path)
        print(f"[PDF] Attempting to load PDF from: {file_path}")

        if not os.path.isfile(file_path):
            print(f"[ERROR] File does NOT exist at path: {file_path}")
            return []

        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path
            print(f"[SUCCESS] Loaded PDF: {len(docs)} pages")
            return docs
        except Exception as e:
            print(f"[ERROR] PDF load error: {e}")
            return []

    # ---------------- Load Website ---------------- #
    def load_website(self, url: str) -> List:
        """Load content from a website URL"""
        try:
            from langchain_community.document_loaders import WebBaseLoader
            print(f"[WEB] Attempting to load website: {url}")
            
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
            print(f"[SUCCESS] Loaded website: {len(docs)} documents")
            return docs
        except Exception as e:
            print(f"[ERROR] Website load error: {e}")
            return []

    # ---------------- Split Documents ---------------- #
    def split_documents(self, documents: List) -> List:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        print(f"[SUCCESS] Split into {len(chunks)} chunks")
        return chunks

    # ---------------- Vector DB ---------------- #
    def create_vector_db(self, documents: List):
        if not documents:
            raise ValueError("[ERROR] No documents to create vector DB")
        chunks = self.split_documents(documents)
        self.documents = chunks
        self.db = FAISS.from_documents(chunks, self.embeddings)
        print(f"[SUCCESS] Vector DB created with {len(chunks)} chunks")

    # ---------------- Query ---------------- #
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG pipeline and return answer with sources"""
        if self.db is None:
            return {"error": "Vector DB not initialized"}
        
        try:
            retriever = self.db.as_retriever(search_kwargs={"k": top_k})
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            result = qa_chain({"query": question})
            
            # Normalize response keys for both OpenAI and Ollama
            answer_text = result.get("result") or result.get("answer") or ""
            source_docs = result.get("source_documents", [])
            
            return {
                "question": question,
                "answer": answer_text if answer_text else "No answer generated. Please try again.",
                "sources": [
                    {"content": doc.page_content[:200], "metadata": doc.metadata}
                    for doc in source_docs
                ] if source_docs else []
            }
        except Exception as e:
            print(f"[ERROR] Query error: {e}")
            return {
                "error": str(e),
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "sources": []
            }

    # ---------------- Clear Data ---------------- #
    def clear(self) -> None:
        """Clear all loaded data"""
        self.db = None
        self.documents = []
        print("[SUCCESS] Cleared all data")

# ---------------- MAIN SCRIPT ---------------- #
if __name__ == "__main__":
    rag = RAGPipeline()

    # Use BlueBridge PDF from Downloads as default, or ask for input
    default_pdf = r"C:\Users\ADMIN\Downloads\BlueBridge_Logical_Reasoning_Shortcuts.pdf"
    pdf_path = default_pdf if os.path.isfile(default_pdf) else input("Enter the full path of your PDF file: ").strip()

    # Debug: show what file will be used
    print(f"📍 Using PDF: {pdf_path}")
    print(f"✅ Exists? {os.path.isfile(pdf_path)}")

    # Load PDF dynamically
    docs = rag.load_pdf(pdf_path)
    if not docs:
        print("❌ No documents loaded. Exiting.")
        exit()

    # Create vector DB and query
    rag.create_vector_db(docs)
    question = input("Enter your query about this document: ").strip()
    res = rag.query(question)
    print("\nAnswer:\n", res.get("answer"))
    print("\nSources:\n", res.get("sources"))