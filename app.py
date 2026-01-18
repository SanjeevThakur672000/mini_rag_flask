import time
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

load_dotenv()

from rag import ingest_document, retrieve, generate_answer


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ingest", methods=["POST"])
def ingest():
    start = time.time()
    text = request.form.get("text", "").strip()

    if len(text) < 5:
        return jsonify({"error": "Text too short"}), 400

    chunks = ingest_document(text, source="user", title="pasted-text")

    return jsonify({
        "message": "Ingested successfully",
        "chunks": len(chunks),
        "time_ms": int((time.time() - start) * 1000)
    })

@app.route("/ask", methods=["POST"])
def ask():
    start = time.time()
    query = request.form.get("query", "").strip()

    if len(query) < 2:
        return jsonify({"error": "Query too short"}), 400

    retrieved = retrieve(query, top_k=8)
    result = generate_answer(query, retrieved)



    sources = []
    for i, s in enumerate(retrieved):
        
        sources.append({
            "id": i + 1,
            "title": s.get("title", ""),
            "source": s.get("source", ""),
            "position": s.get("position", -1),   
            "chunk_id": s.get("chunk_id", -1),
            "score": round(s.get("score", 0), 4),
            "snippet": (s.get("text", "")[:250] + "...") if s.get("text") else ""
        })


    return jsonify({
        "answer": result["answer"],
        "sources": sources,
        "time_ms": int((time.time() - start) * 1000)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
