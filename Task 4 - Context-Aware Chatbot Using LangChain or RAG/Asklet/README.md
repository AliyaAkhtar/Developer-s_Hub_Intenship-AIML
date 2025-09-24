# 📖 Context-Aware Chatbot with PDF Knowledge Base

This project is a **Context-Aware Chatbot** built as part of an internship task.  
It combines a **FastAPI backend** (for handling documents and AI models) with a **Next.js frontend** (for user interaction).  
The chatbot can read PDFs, store their embeddings in a vector database, and answer user queries using conversational context.

## 🚀 Features
- 📂 Upload PDFs and process them into embeddings using FAISS.
- 💬 Context-aware conversation — remembers previous user inputs during the session.
- 🤖 AI-powered answers using LLMs (Groq API + LangChain).
- 🌐 Frontend built with Next.js and React.
- ⚡ Backend built with FastAPI.
- 🔍 Retrieval-Augmented Generation (RAG) for document-based Q&A.

## 🏗️ Tech Stack
- **Frontend**: Next.js 13+, React, Axios, Tailwind CSS  
- **Backend**: FastAPI, LangChain, FAISS, HuggingFace Embeddings, Groq LLM  
- **Database**: Weaviate vector store (cloud)  
- **Other Tools**: dotenv for environment variables  

## 📂 Project Structure
```

project-root/
│── backend/                   # FastAPI backend
│   ├── chatbot.py          
│       ├── app/       
|           ├── api/
|               ├── chatbot/
|                   ├── rag.py  # FastAPI entry point 
│       └── requirement.txt     # Backend dependencies
│   └── .env                    # API keys and configs
│── frontend/ai-chat-docs   # Next.js frontend
│   ├── src/                # Next.js pages/components
│       ├── app/            # Chat UI components
│   ├── package.json        # Frontend dependencies
│   └── README.md           # (this file)


````

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd project-root
````

### 2. Backend Setup (FastAPI)

```bash
cd backend

pip install -r requirements.txt
```

Create `.env` file inside `backend/`:

```env
GROQ_API_KEY=your_groq_api_key_here
WEAVIATE_URL=your_weaviate_url_here
WEAVIATE_API_KEY=your_weaviate_api_key_here
```

Run FastAPI:

```bash
uvicorn chatbot.app.main:app --host 0.0.0.0 --port 8010 --reload
```

### 3. Frontend Setup (Next.js)

```bash
cd frontend
cd ai-chat-docs
npm install
npm run dev
```

Frontend available at: **[http://localhost:3000](http://localhost:3000)**

---

## 🧑‍💻 Usage

1. Upload a PDF via the UI.
   → Backend processes and stores embeddings in Weaviate.
2. Ask a question in the chat.
   → Backend retrieves relevant info and answers with context.
3. Continue asking follow-ups — chatbot remembers past questions.