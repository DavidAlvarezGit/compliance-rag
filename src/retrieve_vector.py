from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "data" / "artifacts"

index = faiss.read_index(str(ARTIFACT_DIR / "faiss.index"))
df = pd.read_parquet(ARTIFACT_DIR / "embedding_metadata.parquet")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def search(query, top_k=5):
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, top_k)
    similarities = scores[0]
    if getattr(index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_L2:
        similarities = -similarities

    results = df.iloc[indices[0]].copy()
    results["similarity"] = similarities

    return results

if __name__ == "__main__":
    query = "menaces pesant sur la croissance"
    results = search(query)

    for _, row in results.iterrows():
        print("=" * 80)
        print(row["doc_id"], f"pp. {row['page_start']}-{row['page_end']}")
        print(row["chunk_text"][:800])
