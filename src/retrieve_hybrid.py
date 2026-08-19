import os
from pathlib import Path
import re
import unicodedata
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from .artifacts import MANIFEST_NAME, validate_artifacts
    from .retrieval_utils import query_references_unknown_year
except ImportError:
    from artifacts import MANIFEST_NAME, validate_artifacts
    from retrieval_utils import query_references_unknown_year

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
MIN_VECTOR_SIMILARITY = float(os.getenv("MIN_VECTOR_SIMILARITY", "0.25"))

@lru_cache(maxsize=1)
def load_chunks_df():
    return pd.read_parquet(CHUNKS_PATH)


# ------------------------------------------------
# BM25 Setup (global for speed)
# ------------------------------------------------
def tokenize(text):
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("’", "'")
    return re.findall(r"[a-z0-9]+", normalized)


@lru_cache(maxsize=1)
def load_bm25():
    df = load_chunks_df()
    corpus = df["chunk_text"].tolist()
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


# ------------------------------------------------
# Vector Setup (global for speed)
# ------------------------------------------------
@lru_cache(maxsize=1)
def load_index():
    return faiss.read_index(str(ARTIFACT_DIR / "faiss.index"))


@lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ------------------------------------------------
# Hybrid Search
# ------------------------------------------------
def hybrid_search(query, top_k=5, max_chunks_per_doc=2):
    df = load_chunks_df()
    if (
        not str(query).strip()
        or top_k <= 0
        or df.empty
        or query_references_unknown_year(query, df)
    ):
        return df.iloc[0:0].copy()
    bm25 = load_bm25()
    index = load_index()
    validate_artifacts(
        index,
        df,
        embedding_model=EMBEDDING_MODEL,
        manifest_path=ARTIFACT_DIR / MANIFEST_NAME,
        metadata_path=ARTIFACT_DIR / "embedding_metadata.parquet",
    )
    model = load_model()
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_max = float(np.max(bm25_scores)) if np.max(bm25_scores) > 0 else 1.0
    bm25_indices = np.argsort(bm25_scores)[::-1][:40]

    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    vector_scores, vector_indices = index.search(query_vec, min(40, len(df)))
    valid = vector_indices[0] >= 0
    vector_ids = vector_indices[0][valid]
    vector_sim = vector_scores[0][valid]
    if getattr(index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_L2:
        vector_sim = -vector_sim

    if vector_sim.size == 0 or float(np.max(vector_sim)) < MIN_VECTOR_SIMILARITY:
        return df.iloc[0:0].copy()

    combined_indices = set(bm25_indices.tolist()) | set(vector_ids.tolist())
    results = df.iloc[list(combined_indices)].copy()

    results["bm25_score"] = results.index.map(lambda i: bm25_scores[i] / bm25_max)
    vector_dict = {idx: sim for idx, sim in zip(vector_ids, vector_sim)}
    vector_floor = min(0.0, float(np.min(vector_sim)))
    results["vector_score"] = results.index.map(lambda i: vector_dict.get(i, vector_floor))

    results["hybrid_score"] = 0.5 * results["bm25_score"] + 0.5 * results["vector_score"]
    results = results.sort_values("hybrid_score", ascending=False)
    if max_chunks_per_doc > 0:
        results["_doc_rank"] = results.groupby("doc_id").cumcount()
        results = results[results["_doc_rank"] < max_chunks_per_doc].drop(columns="_doc_rank")

    return results.head(top_k).copy()
