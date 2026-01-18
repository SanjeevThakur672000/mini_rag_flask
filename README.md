# mini_rag_flask
A lightweight RAG (Retrieval-Augmented Generation) application built with Cohere and Qdrant. Features text/file ingestion, semantic search with reranking, and LLM-generated answers with precise citations.

#  Mini RAG Flask App

A lightweight **Retrieval-Augmented Generation (RAG)** application built with **Flask**, **Cohere**, and **Qdrant**. This project allows users to upload documents or input text, which is then chunked, embedded, and stored in a vector database to answer queries with precise citations.

##  Features

* **RAG Pipeline:** Retrieves relevant context chunks to answer user queries.
* **Citations:** Provides source references for every generated answer.
* **Smart Chunking:** Text is processed via `chunker.py` before embedding.
* **Vector Search:** Utilizes **Qdrant Cloud** for efficient semantic search.
* **Re-ranking:** Uses **Cohere Rerank** to ensure the most relevant chunks are used.

##  Project Structure

```text
mini_rag_flask/
├── app.py           # Main Flask application entry point (API routes)
├── rag.py           # Core RAG logic (Retrieval & Generation functions)
├── chunker.py       # Text processing and chunking utilities
├── requirements.txt # Python dependencies
└── .env             # API keys and configuration (not committed)
