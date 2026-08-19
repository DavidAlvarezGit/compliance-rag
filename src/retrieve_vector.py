import os
from pathlib import Path
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from .artifacts import MANIFEST_NAME, validate_artifacts
except ImportError:
    from artifacts import MANIFEST_NAME, validate_artifacts

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "data" / "artifacts"
load_dotenv(BASE_DIR / ".env")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

@lru_cache(maxsize=1)
def load_index():
    return faiss.read_index(str(ARTIFACT_DIR / "faiss.index"))


@lru_cache(maxsize=1)
def load_df():
    return pd.read_parquet(ARTIFACT_DIR / "embedding_metadata.parquet")


@lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def search(query, top_k=5):
    if not str(query).strip() or top_k <= 0:
        return load_df().iloc[0:0].assign(similarity=pd.Series(dtype=float))
    index = load_index()
    df = load_df()
    validate_artifacts(
        index,
        df,
        embedding_model=EMBEDDING_MODEL,
        manifest_path=ARTIFACT_DIR / MANIFEST_NAME,
        metadata_path=ARTIFACT_DIR / "embedding_metadata.parquet",
    )
    model = load_model()
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, min(top_k, len(df)))
    valid = indices[0] >= 0
    similarities = scores[0][valid]
    if getattr(index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_L2:
        similarities = -similarities

    results = df.iloc[indices[0][valid]].copy()
    results["similarity"] = similarities

    return results

if __name__ == "__main__":
    query = "menaces pesant sur la croissance"
    results = search(query)

    for _, row in results.iterrows():
        print("=" * 80)
        print(row["doc_id"], f"pp. {row['page_start']}-{row['page_end']}")
        print(row["chunk_text"][:800])
