import os
from pathlib import Path
import re
import unicodedata

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"
ARTIFACT_DIR = BASE_DIR / "data" / "artifacts"
load_dotenv(BASE_DIR / ".env")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

df = pd.read_parquet(CHUNKS_PATH)


# ------------------------------------------------
# BM25 Setup (global for speed)
# ------------------------------------------------
def tokenize(text):
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("’", "'")
    return re.findall(r"[a-z0-9]+", normalized)


corpus = df["chunk_text"].tolist()
tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)


# ------------------------------------------------
# Vector Setup (global for speed)
# ------------------------------------------------
index = faiss.read_index(str(ARTIFACT_DIR / "faiss.index"))
model = SentenceTransformer(EMBEDDING_MODEL)


# ------------------------------------------------
# Hybrid Search
# ------------------------------------------------
def hybrid_search(query, top_k=5):
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_indices = np.argsort(bm25_scores)[::-1][:40]

    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    vector_scores, vector_indices = index.search(query_vec, 40)
    vector_sim = vector_scores[0]
    if getattr(index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_L2:
        vector_sim = -vector_sim

    combined_indices = set(bm25_indices.tolist()) | set(vector_indices[0].tolist())
    results = df.iloc[list(combined_indices)].copy()

    results["bm25_score"] = results.index.map(lambda i: bm25_scores[i] / bm25_max)
    vector_dict = {idx: sim for idx, sim in zip(vector_indices[0], vector_sim)}
    results["vector_score"] = results.index.map(lambda i: vector_dict.get(i, 0))

    results["hybrid_score"] = 0.5 * results["bm25_score"] + 0.5 * results["vector_score"]
    results = results.sort_values("hybrid_score", ascending=False)

    return results.head(top_k).copy()
