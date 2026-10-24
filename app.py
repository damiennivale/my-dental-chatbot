"""
MOH CPG RAG System — Flask app
Serves frontend + backend from a single Flask instance

Request flow (matches the documented requirements):
  1. Accept query + mode (Professional / Patient)
  2. Guardrails run BEFORE retrieval (emergency, prompt injection, scope);
     blocked queries never reach retrieval or the LLM
  3. Retrieve the top-5 chunks and generate a grounded answer
  4. Hallucinations are flagged with a warning (inside rag_pipeline)
"""

import os
import sys
import time

# Add backend/ to import path (app.py is at root, backend/ is one level down)
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, BACKEND_DIR)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from rag_engine import get_rag_engine
from guardrails import get_guardrails
from llm import rag_pipeline, warm_up

# Frontend folder
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

rag_engine = get_rag_engine()
guardrails = get_guardrails()
warm_up()  # load the hallucination-check model now, so the first query is fast

VALID_MODES = {"professional", "patient"}
MAX_QUERY_CHARS = 500
TOP_K = 5  # requirement: retrieve the top-5 relevant chunks


# -------- Serve frontend --------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# -------- API --------
@app.route("/api/health", methods=["GET"])
def health_check():
    stats = dict(rag_engine.get_stats())
    # index.html reads "total_chunks"; keep both keys so the UI shows the count
    stats["total_chunks"] = stats.get("num_chunks", 0)
    return jsonify({"status": "healthy", "database": stats})


@app.route("/api/query", methods=["POST"])
def process_query():
    start = time.perf_counter()
    data = request.get_json(silent=True) or {}

    # ---- Accept query + mode selection ----
    query = str(data.get("query", "")).strip()[:MAX_QUERY_CHARS]
    if not query:
        return jsonify({"error": "Empty query"}), 400

    mode = data.get("mode", "professional")
    if mode not in VALID_MODES:
        mode = "professional"

    # ---- Guardrails, pre-retrieval (no LLM involved) ----
    safety = guardrails.process_query(query, mode)
    if safety["blocked"]:
        return jsonify({
            "response": safety["message"],
            "safety_blocked": True,
            "reason": safety["reason"],
            "mode": mode,
            "retrieved_chunks": [],
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        })

    # ---- Retrieve top-5 + generate + hallucination check + format ----
    try:
        answer, chunks = rag_pipeline(query, mode=mode, top_k=TOP_K)
    except Exception:
        app.logger.exception("RAG pipeline failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500

    elapsed = round(time.perf_counter() - start, 3)
    print(f"⏱️  {mode} query answered in {elapsed}s")

    return jsonify({
        "response": answer,
        "safety_blocked": False,
        "mode": mode,
        "retrieved_chunks": chunks,
        "num_results": len(chunks),
        "elapsed_seconds": elapsed,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("🦷 MyDental Chatbot")
    print("=" * 60)
    print("📍 http://localhost:5000")
    print("=" * 60)

    import webbrowser
    import threading
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5000")).start()

    # debug=False and 127.0.0.1: the Werkzeug debugger on an open host
    # lets anyone on your network run code on your machine.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)