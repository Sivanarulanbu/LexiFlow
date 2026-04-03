"""
Advanced Features for RAG Chatbot
- Chat memory with context
- Hybrid search (semantic + keyword)
- Role-based AI assistants
- Source citation
"""

from typing import List, Dict, Tuple
from datetime import datetime
from rag_pipeline import RAGPipeline
import json


class ChatMemory:
    """Manage conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.history: List[Dict] = []
        self.max_history = max_history
    
    def add_turn(self, question: str, answer: str, sources: List) -> None:
        """Add a conversation turn"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": [s.page_content for s in sources],
            "source_count": len(sources)
        })
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self, num_turns: int = 3) -> str:
        """Get recent context for follow-up questions"""
        recent = self.history[-num_turns:]
        context = "Previous conversation:\n"
        
        for turn in recent:
            context += f"Q: {turn['question']}\n"
            context += f"A: {turn['answer'][:200]}...\n\n"
        
        return context
    
    def clear(self) -> None:
        """Clear history"""
        self.history = []
    
    def save(self, filepath: str) -> None:
        """Save conversation to file"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load conversation from file"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)


class HybridSearch:
    """Combine keyword search + semantic search"""
    
    @staticmethod
    def keyword_search(documents: List, query: str, top_k: int = 3) -> List:
        """Simple keyword-based search"""
        query_words = set(query.lower().split())
        scores = []
        
        for doc in documents:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words & doc_words)
            scores.append((doc, overlap))
        
        # Sort by relevance
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scores[:top_k]]
    
    @staticmethod
    def hybrid_search(
        rag: RAGPipeline,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List:
        """Combine semantic + keyword search"""
        
        # Semantic search
        semantic_results = rag.similarity_search(query, k=top_k)
        semantic_docs = [doc for doc, _ in semantic_results]
        
        # Keyword search
        keyword_docs = HybridSearch.keyword_search(rag.documents, query, top_k)
        
        # Combine with weights
        combined = {}
        for i, doc in enumerate(semantic_docs):
            key = id(doc)
            combined[key] = semantic_weight * (1 - i/top_k)
        
        for i, doc in enumerate(keyword_docs):
            key = id(doc)
            if key in combined:
                combined[key] += keyword_weight * (1 - i/top_k)
            else:
                combined[key] = keyword_weight * (1 - i/top_k)
        
        # Sort by combined score
        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]


class RoleBasedAI:
    """Different AI personalities for specific roles"""
    
    SYSTEM_PROMPTS = {
        "HR_Assistant": """You are an HR Assistant. Answer employee questions about:
- Company policies
- Benefits and payroll
- Leave and attendance
- Professional development
Be helpful, professional, and always refer to official policies when needed.""",
        
        "Legal_Assistant": """You are a Legal Assistant. Answer questions about:
- Legal documents
- Contracts and agreements
- Regulations and compliance
- Legal procedures
Provide accurate information and note that this is not legal advice. Recommend consulting an attorney for complex matters.""",
        
        "College_Assistant": """You are a College Assistant. Help students with:
- Course information
- Academic resources
- Exam preparation
- Campus facilities
Be encouraging and supportive in your responses.""",
        
        "Customer_Support": """You are a Customer Support Agent. Help customers with:
- Product information
- Troubleshooting
- Returns and refunds
- General inquiries
Be friendly, empathetic, and solution-focused.""",
        
        "Technical_Assistant": """You are a Technical Assistant. Help users with:
- Technical documentation
- API usage
- Code examples
- System configuration
Provide clear, step-by-step guidance.""",
    }
    
    @staticmethod
    def get_system_prompt(role: str) -> str:
        """Get system prompt for a role"""
        return RoleBasedAI.SYSTEM_PROMPTS.get(role, RoleBasedAI.SYSTEM_PROMPTS["Customer_Support"])
    
    @staticmethod
    def list_roles() -> List[str]:
        """List available roles"""
        return list(RoleBasedAI.SYSTEM_PROMPTS.keys())


class SourceCitation:
    """Handle source citations and attribution"""
    
    @staticmethod
    def format_sources(sources: List, with_page: bool = True) -> str:
        """Format sources for display"""
        formatted = "\n**Sources:**\n"
        
        for i, source in enumerate(sources, 1):
            metadata = source.metadata if hasattr(source, 'metadata') else {}
            page = metadata.get('page', 'N/A')
            source_name = metadata.get('source', 'Document')
            
            if with_page:
                formatted += f"{i}. {source_name} (Page {page})\n"
            else:
                formatted += f"{i}. {source_name}\n"
        
        return formatted
    
    @staticmethod
    def create_citation_html(sources: List) -> str:
        """Create HTML-formatted citations"""
        html = '<div class="citations">'
        
        for i, source in enumerate(sources, 1):
            metadata = source.metadata if hasattr(source, 'metadata') else {}
            page = metadata.get('page', 'N/A')
            
            html += f'<p style="color: #666; font-size: 12px;">'
            html += f'[{i}] Source from Page {page}'
            html += f'</p>'
        
        html += '</div>'
        return html


class DocumentManager:
    """Manage multiple documents and their metadata"""
    
    def __init__(self):
        self.documents: Dict[str, List] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def add_document_group(self, group_name: str, documents: List, metadata: Dict = None) -> None:
        """Add a group of documents"""
        self.documents[group_name] = documents
        self.metadata[group_name] = metadata or {
            "added": datetime.now().isoformat(),
            "doc_count": len(documents)
        }
    
    def get_documents(self, group_name: str = None) -> List:
        """Get documents from a group or all documents"""
        if group_name:
            return self.documents.get(group_name, [])
        
        all_docs = []
        for docs in self.documents.values():
            all_docs.extend(docs)
        return all_docs
    
    def list_groups(self) -> List[str]:
        """List all document groups"""
        return list(self.documents.keys())
    
    def remove_group(self, group_name: str) -> bool:
        """Remove a document group"""
        if group_name in self.documents:
            del self.documents[group_name]
            del self.metadata[group_name]
            return True
        return False


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    # Initialize
    memory = ChatMemory()
    memory.add_turn("What is RAG?", "RAG is Retrieval-Augmented Generation...", [])
    
    print("Chat Memory:", memory.history)
    print("Roles available:", RoleBasedAI.list_roles())
    print("HR System Prompt:", RoleBasedAI.get_system_prompt("HR_Assistant")[:100])
