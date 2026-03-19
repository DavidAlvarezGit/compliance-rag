from __future__ import annotations

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

try:
    from .rag import load_chunks, normalize_query_tokens
except ImportError:
    from rag import load_chunks, normalize_query_tokens


def search(query, top_k=5):
    df = load_chunks()
    tokenized_corpus = [normalize_query_tokens(doc) for doc in df["chunk_text"].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(normalize_query_tokens(query))
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    return pd.DataFrame(results)


if __name__ == "__main__":
    query = "risques pour l’économie suisse"
    results = search(query)
    for _, row in results.iterrows():
        print("=" * 80)
        print(row["doc_id"], f"pp. {row['page_start']}-{row['page_end']}")
        print(row["chunk_text"][:800])
