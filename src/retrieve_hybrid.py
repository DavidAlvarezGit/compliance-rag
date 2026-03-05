from pathlib import Path
import pandas as pd
import numpy as np
import faiss
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"
ARTIFACT_DIR = BASE_DIR / "data" / "artifacts"

df = pd.read_parquet(CHUNKS_PATH)

# ------------------------------------------------
# BM25 Setup (MUST be global)
# ------------------------------------------------
def tokenize(text):
    return text.lower().split()

corpus = df["chunk_text"].tolist()
tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# ------------------------------------------------
# Vector Setup (MUST be global)
# ------------------------------------------------
index = faiss.read_index(str(ARTIFACT_DIR / "faiss.index"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------------------------
# Time Helpers
# ------------------------------------------------
def extract_year(query):
    match = re.search(r"\b20\d{2}\b", query)
    return int(match.group()) if match else None


def detect_relative_time(query):
    q = query.lower()
    TIME_WORDS = [
        "récent", "récemment", "actuel", "actuellement",
        "dernier", "dernière", "derniers", "dernières",
        "aujourd", "ces derniers", "depuis"
    ]
    return any(w in q for w in TIME_WORDS)

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

    # --- Time logic ---
    explicit_year = extract_year(query)
    relative_time = detect_relative_time(query)

    max_year = df["year"].max()
    min_year = df["year"].min()

    if explicit_year:
        filtered = results[results["year"] == explicit_year]
        if not filtered.empty:
            results = filtered

    elif relative_time:
        results["recency_weight"] = (
            (results["year"] - min_year) / (max_year - min_year + 1e-6)
        )
        results["hybrid_score"] += 0.35 * results["recency_weight"]

    results = results.sort_values("hybrid_score", ascending=False)

    # Diversity control
    max_per_doc = 2
    selected = []
    doc_counts = {}

    for _, row in results.iterrows():
        doc_id = row["doc_id"]
        if doc_counts.get(doc_id, 0) < max_per_doc:
            selected.append(row)
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        if len(selected) >= top_k:
            break

    return pd.DataFrame(selected)
