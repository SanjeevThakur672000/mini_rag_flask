# 🧠 Mini RAG System

> A lightweight, production-ready Retrieval-Augmented Generation (RAG) API built with Flask, Cohere, and Qdrant.

**Live URL:** [https://mini-rag-flask.onrender.com](https://mini-rag-flask.onrender.com)  
**Author:** Sanjeev Thakur  

---

##  Project Overview

This project implements a complete **RAG pipeline** designed to answer user queries with high precision by retrieving relevant context from a vector database. Unlike standard chatbots, this system minimizes hallucinations by grounding answers in retrieved data and providing strict **citations** for every claim.

### Key Features
* **End-to-End RAG:** Ingestion, embedding, retrieval, reranking, and generation.
* **Smart Reranking:** Utilizes **Cohere Rerank** to boost retrieval accuracy (fixing the "lost in the middle" phenomenon).
* **Citations:** Every answer includes references to the specific source chunk.
* **Cloud Native:** Uses **Qdrant Cloud** for managed vector storage and **Render** for serverless deployment.

---

##  Architecture

The system follows a standard RAG workflow:

1.  **Ingestion:**
    * Input text is received via the `/ingest` endpoint.
    * Text is processed and split into chunks of 500 characters to ensure optimal embedding performance.
2.  **Embedding:**
    * Chunks are converted into dense vector representations using **Cohere `embed-english-v3.0`** (1024 dimensions).
3.  **Storage:**
    * Vectors and their metadata (original text, source filename) are upserted into **Qdrant Cloud**.
4.  **Retrieval:**
    * User queries are embedded and matched against the database using **Cosine Similarity**.
5.  **Reranking (The "Secret Sauce"):**
    * The top 10 results are passed to **Cohere Rerank**, which uses a cross-encoder to re-score matches based on true semantic relevance, discarding irrelevant noise.
6.  **Generation:**
    * The top 3 reranked chunks are fed into **Cohere Command R+**, which generates the final answer and appends citations.

---

##  Index Configuration (Schema)

This project uses **Qdrant** as the vector store. The collection is configured as follows:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Collection Name** | `mini_rag_docs` | The main index for document chunks. |
| **Vector Size** | `1024` | Matches the output dimension of Cohere Embed v3. |
| **Distance Metric** | `Cosine` | Optimized for semantic similarity. |
| **Payload Schema** | `text` (str) | The raw content of the chunk. |
| | `source` (str) | Filename or origin ID (e.g., "handbook.pdf"). |
| | `chunk_id` (int) | The sequence index of the chunk in the original doc. |

---

##  Setup & Installation

### Prerequisites
* Python 3.9+
* API Keys for **Cohere** and **Qdrant**

### 1. Clone the Repository
```bash
git clone [https://github.com/SanjeevThakur672000/mini_rag_flask.git](https://github.com/SanjeevThakur672000/mini_rag_flask.git)
cd mini_rag_flask
