from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"
ARTIFACT_DIR = BASE_DIR / "data" / "artifacts"

df = pd.read_parquet(CHUNKS_PATH)

# --- BM25 Setup ---
def tokenize(text):
    return text.lower().split()

corpus = df["chunk_text"].tolist()
tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# --- Vector Setup ---
index = faiss.read_index(str(ARTIFACT_DIR / "faiss.index"))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def hybrid_search(query, top_k=5):

    # --- BM25 ---
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:20]

    # --- Vector ---
    query_vec = model.encode([query]).astype("float32")
    distances, vector_indices = index.search(query_vec, 20)

    # --- Combine ---
    combined_indices = set(bm25_indices.tolist()) | set(vector_indices[0].tolist())

    results = df.iloc[list(combined_indices)].copy()

    # Normalize BM25
    results["bm25_score"] = results.index.map(lambda i: bm25_scores[i])
    results["bm25_score"] = results["bm25_score"] / results["bm25_score"].max()

    # Add vector similarity (convert distance → similarity)
    vector_dict = {idx: dist for idx, dist in zip(vector_indices[0], distances[0])}
    results["vector_score"] = results.index.map(lambda i: vector_dict.get(i, np.nan))
    results["vector_score"] = 1 / (1 + results["vector_score"])  # invert distance

    results["vector_score"] = results["vector_score"].fillna(0)

    # Final combined score
    results["hybrid_score"] = 0.5 * results["bm25_score"] + 0.5 * results["vector_score"]

    results = results.sort_values("hybrid_score", ascending=False).head(top_k)

    return results

if __name__ == "__main__":
    query = "menaces pesant sur la croissance"
    results = hybrid_search(query)

    for _, row in results.iterrows():
        print("=" * 80)
        print(row["doc_id"], f"pp. {row['page_start']}-{row['page_end']}")
        print(row["chunk_text"][:700])