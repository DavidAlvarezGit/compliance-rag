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
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = df.iloc[indices[0]].copy()
    results["distance"] = distances[0]

    return results

if __name__ == "__main__":
    query = "menaces pesant sur la croissance"
    results = search(query)

    for _, row in results.iterrows():
        print("=" * 80)
        print(row["doc_id"], f"pp. {row['page_start']}-{row['page_end']}")
        print(row["chunk_text"][:800])