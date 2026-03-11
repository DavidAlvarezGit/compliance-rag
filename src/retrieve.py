from pathlib import Path
import re
import unicodedata

import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"

df = pd.read_parquet(CHUNKS_PATH)

def tokenize(text):
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("’", "'")
    return re.findall(r"[a-z0-9]+", normalized)

corpus = df["chunk_text"].tolist()
tokenized_corpus = [tokenize(doc) for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

def search(query, top_k=5):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]

    return results

if __name__ == "__main__":
    query = "risques pour l’économie suisse"
    results = search(query)

    results = search(query)
    
    for i, row in results.iterrows():
        print("=" * 80)
        print(row["doc_id"], f"pp. {row['page_start']}-{row['page_end']}")
        print(row["chunk_text"][:800])
