# 💬 Conversational RAG Q&A with PDFs

This project is a **ChatGPT-style conversational assistant** that lets you upload PDFs and chat with them.  
Each chat has its own memory and its own documents, so you can maintain multiple parallel conversations with different contexts.  

Built with **Streamlit**, **LangChain**, **ChromaDB**, **Groq LLMs**, and **PDF ingestion pipelines**.

---

## ✨ Features

- 🗂️ **Multi-chat support**  
  - Sidebar shows all your chats.  
  - Each chat has its own history and memory.  
  - Chats are auto-titled based on the first exchange.  

- 📚 **Upload PDFs**  
  - Upload one or multiple PDFs per chat.  
  - Automatically chunked, embedded, and stored in **ChromaDB**.  
  - Chatbot answers are grounded in these documents.  

- 🧠 **Conversational memory**  
  - Remembers what you said earlier in each chat.  
  - Keeps context across multiple turns.  

- 🔄 **Clear controls**  
  - Reset chat history.  
  - Clear all PDFs (removes files + embeddings + retriever).  

- 🎨 **Modern UI**  
  - ChatGPT-like sidebar.  
  - Collapsible, minimal PDF uploader.  
  - Chat bubbles for user/bot messages.  

- ⚡ **LLM powered by Groq**  
  - Fast, low-latency LLM inference using Groq API.  

---

