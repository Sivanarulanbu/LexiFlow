# 🤖 LexiFlow - Intelligent AI RAG Engine

**A production-ready LLM + RAG system for PDF/Website knowledge assistants.**

---

## 🎯 Project Overview
This project is an enterprise-grade **RAG (Retrieval-Augmented Generation)** chatbot developed to transform static documents (PDFs) and websites into interactive, AI-powered knowledge bases. It addresses the common LLM problem of "hallucinations" by grounding every response in specific, retrieved context.

### ✅ Key Capabilities
*   **Multi-Source Knowledge:** Load information from multiple PDF files and website URLs.
*   **Intelligent Text Processing:** Automatic semantic chunking (500 tokens, 50 overlap) preserves context.
*   **High-Speed Retrieval:** Powered by **FAISS** (Facebook AI Similarity Search) for instant semantic lookups.
*   **LLM Intelligence:** Integrates with OpenAI's GPT-3.5/4/4-Turbo for accurate answer generation.
*   **Source Citations:** Automatically cites sources with page numbers to ensure transparency.
*   **Conversation Memory:** Maintains context across multiple turns of dialogue.
*   **Role-Based AI:** Pre-configured personalities (HR, Legal, College, Support, etc.).
*   **Persistent Vector DB:** Save and reload knowledge bases to avoid reprocessing.
*   **Modern UI:** A clean, responsive Streamlit interface for easy interaction.

---

## 🚀 Quick Start Guide

### 1️⃣ Get Your OpenAI API Key
1.  Visit [OpenAI API Keys](https://platform.openai.com/api-keys).
2.  Create a "Secret Key" and copy it safely.

### 2️⃣ Installation & Setup
```bash
# 1. Clone/Download the project and navigate to the directory
cd rag_chatbot

# 2. Install required Python dependencies
pip install -r requirements.txt

# 3. Configure environment variables
# Copy .env.example to .env and paste your API key
# Example: OPENAI_API_KEY=sk-your-key-here
```

### 3️⃣ Launch the Application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 🏗️ Technical Architecture

### The RAG Workflow
```
[Documents/URLs] → [Load & Parse] → [Semantic Chunking] → [Embedding Generation] → [FAISS Vector DB]
                                                                                      ↓
[User Question]  → [Question Embedding] → [Similarity Search] → [Context Retrieval] → [LLM (GPT)]
                                                                                      ↓
                                                                             [Answer + Sources]
```

### Core Components
1.  **`app.py` (UI Layer):** Handles the Streamlit interface, file uploads, and session management.
2.  **`rag_pipeline.py` (Orchestration):** The "brain" that manages loading, chunking, vector storage, and querying.
3.  **`advanced_features.py` (Utility):** Implements memory, hybrid search, role-based prompts, and citation formatting.

### Technical Stack
| Layer | Technology |
|---|---|
| **Interface** | Streamlit, Vanilla CSS |
| **Orchestration** | LangChain |
| **LLM / Embeddings** | OpenAI (GPT-3.5/4, text-embedding-3-small) |
| **Vector Database** | FAISS (In-memory/Local) |
| **Parsing** | PyPDF (PDFs), BeautifulSoup (Websites) |

---

## 📖 Usage Instructions

### 1. Training the AI (Upload & Learn Tab)
*   **PDFs:** Drag and drop files to the uploader and click **"Process PDF(s)"**.
*   **Websites:** Enter a full URL (e.g., `https://docs.example.com`) and click **"Load Website"**.
*   **Status Check:** The sidebar will confirm once the "RAG Pipeline" is initialized.

### 2. Chatting (Chat Tab)
*   Type your question in the text area.
*   Use the **"Number of source documents"** slider to adjust how much context the AI uses.
*   Click **"Ask Question"** to see the answer and expandable source citations.

### 3. Database Management (Advanced Tab)
*   **Save DB:** Persist your currently loaded knowledge to a folder (default: `./vector_db`).
*   **Load DB:** Load a previously saved database to chat without re-uploading documents.
*   **Insights:** View conversation history and basic usage statistics.

---

## ⚙️ Configuration & Optimization

### Tuning "Chunk" Settings (Sidebar)
*   **Chunk Size (Default: 500):** Larger chunks provide more context but may dilute search precision.
*   **Chunk Overlap (Default: 50):** Ensures transitions between chunks aren't lost, maintaining flow.

### Model Selection
*   **gpt-3.5-turbo:** Recommended for most tasks. Fast and cost-effective.
*   **gpt-4:** Higher accuracy and better reasoning for complex documents.

---

## 🌍 Deployment Options

### Streamlit Cloud (Fastest)
1.  Push your code to a GitHub repository (ensure `.env` and `vector_db` are in `.gitignore`).
2.  Connect your repo to [Streamlit Cloud](https://streamlit.io/cloud).
3.  Add your `OPENAI_API_KEY` to the app "Secrets" settings.

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

---

## 🛡️ Best Practices & Security
*   **Never Commit Keys:** Ensure your `.env` file is excluded from version control via `.gitignore`.
*   **Save Progress:** Regularly use the "Save DB" feature for large knowledge bases.
*   **Verify Answers:** Always cross-reference AI answers with the provided **Source Citations**.
*   **Clean Data:** For best results, use text-heavy PDFs and clear, public website URLs.

---

## 🐛 Troubleshooting
| Issue | Potential Solution |
|---|---|
| **API Key Missing** | Ensure `.env` exists and contains `OPENAI_API_KEY`. |
| **No "Ask" Button** | Ensure you have uploaded and processed at least one document. |
| **Slow Performance** | Try reducing the "Number of sources" slider or switching to GPT-3.5. |
| **Import Errors** | Run `pip install --upgrade -r requirements.txt`. |

---

**Built for AI Engineering Excellence | 2026**
