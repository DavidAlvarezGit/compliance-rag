from __future__ import annotations

import faiss
import pandas as pd

try:
    from .rag import load_resources
except ImportError:
    from rag import load_resources


def search(query, top_k=5):
    resources = load_resources()
    query_vec = resources.embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = resources.index.search(query_vec, top_k)
    similarities = scores[0]
    if getattr(resources.index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_L2:
        similarities = -similarities

    results = resources.chunks_df.iloc[indices[0]].copy()
    results["similarity"] = similarities
    return pd.DataFrame(results)


if __name__ == "__main__":
    query = "menaces pesant sur la croissance"
    results = search(query)
    for _, row in results.iterrows():
        print("=" * 80)
        print(row["doc_id"], f"pp. {row['page_start']}-{row['page_end']}")
        print(row["chunk_text"][:800])
