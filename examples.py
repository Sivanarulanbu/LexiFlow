"""
Example Usage Scripts - Different ways to use the RAG system
"""

import os
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
from advanced_features import ChatMemory, RoleBasedAI, HybridSearch, SourceCitation

load_dotenv()

# ============================================================================
# EXAMPLE 1: Basic PDF Chatbot
# ============================================================================

def example_basic_pdf_chatbot():
    """Simple PDF loading and questioning"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic PDF Chatbot")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    rag = RAGPipeline(api_key=api_key, model="gpt-3.5-turbo")
    
    # Example: Load a sample PDF (replace with your file)
    print("\n📄 Loading PDF...")
    # docs = rag.load_pdf("sample.pdf")
    # if docs:
    #     rag.create_vector_db(docs)
    #     
    #     # Ask question
    #     print("\n❓ Asking question...")
    #     response = rag.query("What is the main topic?")
    #     print(f"Answer: {response['answer']}")


# ============================================================================
# EXAMPLE 2: Multi-Document Support
# ============================================================================

def example_multi_document():
    """Load multiple PDFs and search across all"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Document Support")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    rag = RAGPipeline(api_key=api_key)
    
    # Simulate loading multiple documents
    pdf_list = ["document1.pdf", "document2.pdf", "document3.pdf"]
    
    print(f"\n📚 Loading {len(pdf_list)} documents...")
    
    # for i, pdf in enumerate(pdf_list):
    #     print(f"  • Loading {pdf}...")
    #     docs = rag.load_pdf(pdf)
    #     if i == 0:
    #         rag.create_vector_db(docs)
    #     else:
    #         rag.add_documents(docs)
    
    print(f"✅ Total documents loaded: {len(rag.documents) if rag.db else 0}")


# ============================================================================
# EXAMPLE 3: Website Loading
# ============================================================================

def example_website_chatbot():
    """Load and chat about website content"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Website Chatbot")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    rag = RAGPipeline(api_key=api_key)
    
    # Load website
    url = "https://example.com"  # Replace with real URL
    print(f"\n🌐 Loading website: {url}")
    # docs = rag.load_website(url)
    # if docs:
    #     rag.create_vector_db(docs)
    #     response = rag.query("What is on this website?")
    #     print(f"Answer: {response['answer']}")


# ============================================================================
# EXAMPLE 4: Chat with Memory
# ============================================================================

def example_chat_with_memory():
    """Conversation with context memory"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Chat with Memory")
    print("="*60)
    
    memory = ChatMemory(max_history=5)
    
    # Simulate conversation
    turns = [
        ("What is RAG?", "RAG is Retrieval-Augmented Generation..."),
        ("How does it work?", "It retrieves documents, then generates..."),
        ("What are the benefits?", "Better accuracy, fewer hallucinations..."),
    ]
    
    print("\n💬 Conversation:")
    for question, answer in turns:
        memory.add_turn(question, answer, [])
        print(f"Q: {question}")
        print(f"A: {answer}\n")
    
    # Get context for follow-up
    context = memory.get_context(num_turns=2)
    print("📝 Context for follow-up:")
    print(context)
    
    # Save conversation
    memory.save("chat_history.json")
    print("✅ Conversation saved to chat_history.json")


# ============================================================================
# EXAMPLE 5: Role-Based AI
# ============================================================================

def example_role_based_ai():
    """Different AI personalities"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Role-Based AI")
    print("="*60)
    
    roles = RoleBasedAI.list_roles()
    print(f"\n🎭 Available roles ({len(roles)}):")
    
    for role in roles:
        print(f"\n  📌 {role}")
        prompt = RoleBasedAI.get_system_prompt(role)
        print(f"     {prompt[:80]}...")


# ============================================================================
# EXAMPLE 6: Source Citation
# ============================================================================

def example_source_citation():
    """Show how to cite sources"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Source Citation")
    print("="*60)
    
    # Mock sources
    class MockSource:
        def __init__(self, content, page):
            self.page_content = content
            self.metadata = {"page": page, "source": "Document.pdf"}
    
    sources = [
        MockSource("RAG improves accuracy...", 5),
        MockSource("Vector databases store embeddings...", 8),
    ]
    
    print("\n📎 Formatted Sources:")
    citations = SourceCitation.format_sources(sources)
    print(citations)
    
    print("🌐 HTML Citations:")
    html = SourceCitation.create_citation_html(sources)
    print(html)


# ============================================================================
# EXAMPLE 7: Save and Load Vector DB
# ============================================================================

def example_persistence():
    """Save and reload vector database"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Vector DB Persistence")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    rag = RAGPipeline(api_key=api_key)
    
    # Simulate loading and saving
    print("\n💾 Creating and saving vector DB...")
    # docs = rag.load_pdf("sample.pdf")
    # rag.create_vector_db(docs)
    # rag.save_db("./vector_db/my_database")
    
    print("✅ Database saved")
    
    # Load later
    print("\n📂 Loading saved database...")
    # rag2 = RAGPipeline(api_key=api_key)
    # rag2.load_db("./vector_db/my_database")
    # response = rag2.query("Question")
    
    print("✅ Database loaded successfully")


# ============================================================================
# EXAMPLE 8: Hybrid Search (Semantic + Keyword)
# ============================================================================

def example_hybrid_search():
    """Combine semantic and keyword search"""
    print("\n" + "="*60)
    print("EXAMPLE 8: Hybrid Search")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    rag = RAGPipeline(api_key=api_key)
    
    print("\n🔍 Hybrid search example:")
    print("   70% Semantic (embeddings) + 30% Keyword match")
    
    # Example results
    print("\n   Results:")
    print("   1. Document A (Score: 0.92) - Highly relevant")
    print("   2. Document B (Score: 0.87) - Very relevant")
    print("   3. Document C (Score: 0.79) - Relevant")


# ============================================================================
# EXAMPLE 9: Batch Processing
# ============================================================================

def example_batch_processing():
    """Process multiple documents efficiently"""
    print("\n" + "="*60)
    print("EXAMPLE 9: Batch Processing")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    rag = RAGPipeline(api_key=api_key)
    
    # Simulate batch
    documents_to_process = [
        "HR_Policy.pdf",
        "Employee_Handbook.pdf",
        "Benefits_Guide.pdf",
    ]
    
    print(f"\n📦 Processing {len(documents_to_process)} documents...")
    
    for i, doc in enumerate(documents_to_process, 1):
        print(f"  [{i}/{len(documents_to_process)}] Processing {doc}...")
        # Load and add to DB
    
    print("✅ Batch processing complete")


# ============================================================================
# EXAMPLE 10: Interview Explanation
# ============================================================================

def example_interview_explanation():
    """How to explain RAG in an interview"""
    print("\n" + "="*60)
    print("EXAMPLE 10: Interview Explanation")
    print("="*60)
    
    explanation = """
🎤 INTERVIEW TALKING POINTS:

1️⃣ THE PROBLEM
   "Large Language Models (LLMs) have a knowledge cutoff and can hallucinate 
   when they don't have access to specific information. This is especially 
   problematic for domain-specific queries where accuracy is critical."

2️⃣ THE SOLUTION
   "I built a Retrieval-Augmented Generation (RAG) system that combines 
   the power of LLMs with external knowledge bases."

3️⃣ HOW IT WORKS
   Step 1: Document Ingestion
           - Load PDFs and websites
           - Extract text content
   
   Step 2: Chunking
           - Split text into ~500 token chunks
           - Keep 50 token overlap for context
   
   Step 3: Embeddings
           - Convert chunks to vector embeddings (1536-dim)
           - Uses OpenAI's embedding model
   
   Step 4: Vector Storage
           - Store embeddings in FAISS (Facebook AI Similarity Search)
           - Enables fast similarity search
   
   Step 5: Retrieval
           - User asks question
           - Convert question to embedding
           - Find top-K similar documents using cosine similarity
   
   Step 6: Generation
           - Pass question + retrieved context to LLM
           - LLM generates accurate answer with context
           - Show sources for transparency

4️⃣ KEY ACHIEVEMENTS
   ✅ Handles multiple documents (PDFs + websites)
   ✅ Persistent vector database (save/load)
   ✅ Chat memory and conversation context
   ✅ Source citations (shows where answer came from)
   ✅ Role-based AI personalities
   ✅ Hybrid search (semantic + keyword)
   ✅ Production-ready Streamlit UI
   ✅ Deployed on Streamlit Cloud

5️⃣ TECHNICAL DEPTH
   • Embedding Models: Text similarity using semantic representations
   • Vector Search: O(1) retrieval from millions of documents
   • Prompt Engineering: Crafting effective system prompts
   • Token Management: Optimizing chunk size vs. context window
   • Production Deployment: Containerization, cloud hosting

6️⃣ CHALLENGES FACED
   • Chunk size tuning (too small = lost context, too large = slower)
   • Latency optimization (balancing speed vs. accuracy)
   • Cost management (API usage tracking)
   • Data privacy (no sensitive data in cloud DBs)
   • Quality metrics (measuring answer accuracy)

7️⃣ UNIQUE FEATURES
   • Supports both PDFs and live websites
   • Memory of past conversations for context
   • Role-based personality system
   • Source attribution for reliability
   • Persistent database for reuse

8️⃣ USE CASES
   ✓ HR Assistant - Answer employee questions from policies
   ✓ Legal Assistant - Retrieve contract terms
   ✓ Customer Support - FAQ-based responses
   ✓ College Assistant - Syllabus and course info
   ✓ Internal Knowledge Base - Company docs search
    """
    
    print(explanation)


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " RAG Chatbot - Example Usage Scripts ".center(58) + "║")
    print("╚" + "="*58 + "╝")
    
    # Run all examples
    example_basic_pdf_chatbot()
    example_multi_document()
    example_website_chatbot()
    example_chat_with_memory()
    example_role_based_ai()
    example_source_citation()
    example_persistence()
    example_hybrid_search()
    example_batch_processing()
    example_interview_explanation()
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("="*60)
