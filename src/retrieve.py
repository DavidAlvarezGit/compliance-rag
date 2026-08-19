from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"


def tokenize(text: object) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"[a-z0-9]+", normalized)


@lru_cache(maxsize=1)
def load_resources() -> tuple[pd.DataFrame, BM25Okapi]:
    chunks = pd.read_parquet(CHUNKS_PATH)
    corpus = [tokenize(text) for text in chunks["chunk_text"].tolist()]
    return chunks, BM25Okapi(corpus)


def search(query: str, top_k: int = 5) -> pd.DataFrame:
    chunks, bm25 = load_resources()
    if not query.strip() or top_k <= 0 or chunks.empty:
        return chunks.iloc[0:0].assign(score=pd.Series(dtype=float))
    scores = np.asarray(bm25.get_scores(tokenize(query)), dtype=float)
    limit = min(top_k, len(chunks))
    indices = np.argsort(scores)[::-1][:limit]
    results = chunks.iloc[indices].copy()
    results["score"] = scores[indices]
    return results


if __name__ == "__main__":
    for _, row in search("risques pour l’économie suisse").iterrows():
        print("=" * 80)
        print(row["doc_id"], f"pp. {row['page_start']}-{row['page_end']}")
        print(row["chunk_text"][:800])
