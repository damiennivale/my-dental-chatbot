"""
rag_engine.py — Hybrid retrieval (ChromaDB dense + BM25 sparse)

Changes from the FYP1 version:
  - Reciprocal Rank Fusion (RRF) instead of adding raw cosine + raw BM25 scores
    (they are on different scales, so BM25 used to dominate the ranking)
  - Chunks are tracked by index/ID, not by their text (duplicate text no longer collides)
  - "similarity" is the real cosine similarity, not a score divided by the best score
  - Off-topic gate: if nothing is similar enough, return [] instead of random chunks
  - Better BM25 tokenizer (strips punctuation)

IMPORTANT: the Chroma collection must use cosine distance.
Re-run ingest.py after adding  metadata={"hnsw:space": "cosine"}  to create_collection.
"""

import os
import re
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi


# ---------- Paths ----------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "cpg_guidelines"

# Minimum cosine similarity of the best dense match to accept a query.
# Placeholder: tune it on your own on-topic and off-topic test questions.
MIN_DENSE_SIM = 0.45


def _tokenize(text: str):
    """Lowercase word tokens, punctuation removed ('caries,' -> 'caries')."""
    return re.findall(r"[a-z0-9]+", text.lower())


class RAGEngine:
    def __init__(self):
        print("🔧 Loading RAG engine...")

        # 1. Embedding function
        print("   → Loading BGE embedding model...")
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        # 2. Open ChromaDB
        print(f"   → Opening ChromaDB at {CHROMA_PATH}")
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
        )
        print(f"   → Collection loaded ({self.collection.count()} chunks)")

        # Safety check: similarity maths below assumes cosine distance
        space = (self.collection.metadata or {}).get("hnsw:space")
        if space != "cosine":
            print(
                "   ⚠️ WARNING: collection distance is "
                f"'{space or 'l2 (default)'}', not 'cosine'. Similarity scores "
                "will be wrong. Add metadata={'hnsw:space': 'cosine'} to "
                "create_collection in ingest.py and re-run it."
            )

        # 3. Fetch all chunks for BM25
        data = self.collection.get(include=["documents", "metadatas"])
        self.all_ids = data["ids"]
        self.all_chunks = data["documents"]
        self.all_metadatas = data["metadatas"]
        self.id_to_idx = {cid: i for i, cid in enumerate(self.all_ids)}
        self.last_max_dense_sim = 0.0   # gate statistic of the most recent search

        # 4. Build BM25 index
        print("   → Building BM25 index...")
        self.bm25 = BM25Okapi([_tokenize(c) for c in self.all_chunks])

        print("✅ RAG engine ready.\n")

    # ------------------------------------------------------------------
    def _cosine_for(self, idx_list, q_vec):
        """Real cosine similarity between the query and the given chunks."""
        if not idx_list:
            return {}
        got = self.collection.get(
            ids=[self.all_ids[i] for i in idx_list], include=["embeddings"]
        )
        out = {}
        q_norm = np.linalg.norm(q_vec) or 1.0
        for cid, emb in zip(got["ids"], got["embeddings"]):
            v = np.asarray(emb, dtype=float)
            denom = (np.linalg.norm(v) or 1.0) * q_norm
            out[self.id_to_idx[cid]] = float(np.dot(v, q_vec) / denom)
        return out

    def search(
        self,
        query: str,
        k: int = 5,
        dense_weight: float = 0.6,
        rrf_k: int = 60,
        min_dense_sim: float = MIN_DENSE_SIM,
    ):
        """
        Hybrid retrieval with Reciprocal Rank Fusion.
        Returns list of dicts: {text, metadata, similarity, fusion_score}
        similarity = real cosine similarity between query and chunk.
        Returns [] if the query is not similar enough to anything.
        """
        n = k * 4  # candidates taken from each retriever

        # --- Dense (ChromaDB) ---
        q_vec = np.asarray(self.embed_fn([query])[0], dtype=float)
        d = self.collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=min(n, len(self.all_ids)),
            include=["distances"],
        )
        dense_idx = [self.id_to_idx[cid] for cid in d["ids"][0]]
        # cosine distance = 1 - cosine similarity
        dense_sim = {
            i: 1.0 - dist for i, dist in zip(dense_idx, d["distances"][0])
        }

        # Remember the gate statistic (evaluate.py uses it to tune MIN_DENSE_SIM)
        self.last_max_dense_sim = max(dense_sim.values()) if dense_sim else 0.0

        # --- Off-topic gate on the real cosine similarity ---
        if not dense_sim or self.last_max_dense_sim < min_dense_sim:
            return []

        # --- Sparse (BM25) ---
        scores = self.bm25.get_scores(_tokenize(query))
        bm25_idx = (
            [int(i) for i in np.argsort(scores)[::-1][:n] if scores[i] > 0]
            if scores.any()
            else []
        )

        # --- Reciprocal Rank Fusion (uses ranks, so scales don't matter) ---
        fused = {}
        for rank, i in enumerate(dense_idx):
            fused[i] = fused.get(i, 0.0) + dense_weight / (rrf_k + rank + 1)
        for rank, i in enumerate(bm25_idx):
            fused[i] = fused.get(i, 0.0) + (1 - dense_weight) / (rrf_k + rank + 1)

        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

        # Real cosine for chunks that only BM25 found
        missing = [i for i, _ in top if i not in dense_sim]
        dense_sim.update(self._cosine_for(missing, q_vec))

        return [
            {
                "text": self.all_chunks[i],
                "metadata": self.all_metadatas[i],
                "similarity": float(dense_sim.get(i, 0.0)),
                "fusion_score": float(score),
            }
            for i, score in top
        ]

    def get_stats(self):
        return {
            "num_chunks": self.collection.count(),
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
        }


# ---------- Singleton ----------
_engine = None


def get_rag_engine():
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine