from pathlib import Path
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"

df = pd.read_parquet(CHUNKS_PATH)

# Tokenize simply by whitespace (good enough for MVP)
def tokenize(text):
    return text.lower().split()

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