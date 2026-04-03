import os
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path

class RAGPipeline:
    def __init__(self):
        """Initialize RAG pipeline with Ollama"""
        self.model_name = os.getenv("MODEL", "mistral")
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        
        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model=self.model_name,
            base_url="http://localhost:11434",
            temperature=0.7
        )
        
        # Initialize embeddings using HuggingFace
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Load or create vector store
        self.vectorstore = self._load_or_create_vectorstore()
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Build the chain using LCEL
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _load_or_create_vectorstore(self):
        """Load existing vector store or create new one"""
        if Path(self.vector_db_path).exists():
            print(f"Loading existing vector store from {self.vector_db_path}")
            return FAISS.load_local(self.vector_db_path, self.embeddings)
        else:
            print(f"Creating new vector store")
            # Create empty vector store
            from langchain_community.docstore import InMemoryDocstore
            from langchain_core.documents import Document
            import faiss
            
            # Create FAISS index
            dimension = 384  # all-MiniLM-L6-v2 dimension
            index = faiss.IndexFlatL2(dimension)
            
            vectorstore = FAISS(
                embedding_function=self.embeddings.embed_query,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            return vectorstore
    
    def add_documents(self, documents):
        """Add documents to the vector store"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        split_docs = text_splitter.split_documents(documents)
        self.vectorstore.add_documents(split_docs)
        self.vectorstore.save_local(self.vector_db_path)
        print(f"Added {len(split_docs)} document chunks to vector store")
    
    def query(self, question: str) -> str:
        """Query the RAG pipeline"""
        try:
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower() or "11434" in error_msg:
                raise ConnectionError(
                    f"Cannot connect to Ollama on localhost:11434. "
                    f"Make sure Ollama is running: `ollama serve`"
                )
            else:
                raise Exception(f"Query failed: {error_msg}")
    
    @staticmethod
    def _format_docs(docs):
        """Format documents for the prompt"""
        return "\n\n".join(doc.page_content for doc in docs)
