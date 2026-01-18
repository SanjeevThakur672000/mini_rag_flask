import os
import uuid
from typing import List, Dict, Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import cohere
from chunker import chunk_text

load_dotenv()

# ---------------------------
# ENV VARIABLES
# ---------------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mini_rag_docs")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY missing in .env")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL missing in .env")

# ---------------------------
# CLIENTS
# ---------------------------
co = cohere.Client(COHERE_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
)

# ---------------------------
# MODELS
# ---------------------------
EMBED_MODEL = "embed-english-v3.0"
GEN_MODEL = "command-r-08-2024"


# ---------------------------
# EMBEDDINGS
# ---------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = co.embed(
        texts=texts,
        model=EMBED_MODEL,
        input_type="search_document",
    )
    return resp.embeddings


def embed_query(query: str) -> List[float]:
    resp = co.embed(
        texts=[query],
        model=EMBED_MODEL,
        input_type="search_query",
    )
    return resp.embeddings[0]


# ---------------------------
# QDRANT COLLECTION
# ---------------------------
def ensure_collection(vector_size: int):
    collections = qdrant.get_collections().collections
    names = [c.name for c in collections]

    if QDRANT_COLLECTION not in names:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


# ---------------------------
# INGEST
# ---------------------------
def ingest_document(text: str, source: str = "user", title: str = "pasted-text"):
    chunks = chunk_text(text, chunk_size=500, overlap=80)

    if not chunks:
        return []

    # Create collection using correct embedding dimension
    sample_vec = embed_texts([chunks[0]])[0]
    ensure_collection(vector_size=len(sample_vec))

    embeddings = embed_texts(chunks)

    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "chunk_id": i,
                    "position": i,
                    "source": source,
                    "title": title,
                    "text": chunk,
                },
            )
        )

    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return chunks


# ---------------------------
# RETRIEVE
# ---------------------------
def retrieve(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    qvec = embed_query(query)

    res = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=qvec,
        limit=top_k
    ).points

    docs = []
    seen_texts = set()

    for r in res:
        payload = r.payload or {}
        text = payload.get("text", "")

        # remove duplicates
        if text in seen_texts:
            continue
        seen_texts.add(text)

        docs.append(
            {
                "score": float(r.score),
                "text": text,
                "title": payload.get("title", ""),
                "source": payload.get("source", ""),
                "chunk_id": payload.get("chunk_id", -1),
                "position": payload.get("position", payload.get("chunk_id", -1)),
            }
        )

    return docs

# ---------------------------
# GENERATE ANSWER
# ---------------------------
def generate_answer(query: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not retrieved:
        return {
            "answer": "No relevant chunks found. Please ingest some text first.",
        }

    context_blocks = []
    for i, d in enumerate(retrieved, start=1):
        context_blocks.append(f"[{i}] {d['text']}")

    context = "\n\n".join(context_blocks)

    message = f"""
You are a helpful RAG assistant.
Answer the question ONLY using the context.
If context is insufficient, say: "Not enough information in the provided text."

Context:
{context}

Question:
{query}

Rules:
- Give short clear answer.
- Add citations like [1], [2] based on used chunks.
"""

    resp = co.chat(
        model=GEN_MODEL,
        message=message,
        temperature=0.2,
    )

    return {
        "answer": resp.text.strip(),
    }
