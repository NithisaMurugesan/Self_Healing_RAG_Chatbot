# 🤖 Self-Healing RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask natural language questions about their contents. The system uses hybrid retrieval, query rewriting, reranking, faithfulness evaluation, and automatic answer regeneration to improve response accuracy and reduce hallucinations. Built using LangChain, ChromaDB, BM25, Hugging Face models, Groq, and Streamlit.
---

## 🚀 Features

* 📄 Dynamic PDF Upload and Processing
* 🔍 Hybrid Retrieval (BM25 + Vector Search)
* 🧠 Query Rewriting for Improved Retrieval
* 🎯 Cross-Encoder Reranking
* 📚 Source Citation Generation
* ✅ Faithfulness Evaluation
* 🔄 Self-Healing Answer Regeneration
* 📊 Latency Monitoring & Observability
* 💬 Interactive Streamlit Interface

---

## 🏗️ System Architecture

```text
PDF Upload
      ↓
Document Loading
      ↓
Chunking
      ↓
Embedding Generation
      ↓
Chroma Vector Database
      ↓
Query Rewriting
      ↓
Hybrid Retrieval
      ↓
Cross-Encoder Reranking
      ↓
Context Construction
      ↓
LLM Generation
      ↓
Faithfulness Evaluation
      ↓
Self-Healing Regeneration
      ↓
Final Response with Citations
```

---

## 🖼️ Application Screenshots

### Main Interface

![Main Interface](ss/interface.png)

### Monitoring & Evaluation Dashboard

![Monitoring Dashboard](ss/monitoring.png)

---

## ⚙️ Technology Stack

### Frontend

* Streamlit

### Document Processing

* PyPDFLoader
* RecursiveCharacterTextSplitter

### Embeddings

* HuggingFace Embeddings
* all-MiniLM-L6-v2

### Vector Database

* ChromaDB

### Retrieval

* BM25 Retriever
* Ensemble Retriever

### Reranking

* BAAI/bge-reranker-base

### LLM

* Qwen3-32B
* Groq API

### Framework

* LangChain

---

## 🔍 Key Components

### Query Rewriting

User questions are automatically rewritten into retrieval-friendly queries before document search, improving retrieval accuracy and relevance.

### Hybrid Retrieval

The system combines:

* Semantic Search using Vector Embeddings
* Keyword Search using BM25

to improve recall and retrieval quality.

### Cross-Encoder Reranking

Retrieved chunks are reranked using BGE Reranker to select only the most relevant context before answer generation.

### Faithfulness Evaluation

A secondary LLM evaluates whether generated answers are grounded in the retrieved context and assigns a faithfulness score between 0 and 100.

### Self-Healing Pipeline

If the faithfulness score falls below the configured threshold:

1. Additional context is retrieved
2. The answer is regenerated
3. The response is re-evaluated

This creates a self-correcting retrieval and generation loop.

---

## 📊 Monitoring & Observability

The application tracks:

* Query Rewrite Time
* Retrieval Time
* Generation Time
* Total Response Latency
* Faithfulness Score
* Retry Count
* Rewritten Query
* Evaluation Reasoning

These metrics help monitor system performance and answer quality.

---

## ▶️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Configure environment variables:

```env
GROQ_API_KEY=your_api_key
```

Run the application:

```bash
streamlit run app.py
```

## 👨‍💻 Author

**Nithisa Murugesan**
