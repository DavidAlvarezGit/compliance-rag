from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.0"))


@lru_cache(maxsize=1)
def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def score_passages(
    query: str,
    passages: Sequence[str],
    *,
    model: CrossEncoder | None = None,
) -> np.ndarray:
    if not passages:
        return np.array([], dtype=float)
    reranker = model or load_reranker()
    pairs = [(query, passage) for passage in passages]
    return np.asarray(reranker.predict(pairs, show_progress_bar=False), dtype=float).reshape(-1)


def has_relevant_passage(scores: Sequence[float], threshold: float = MIN_RERANK_SCORE) -> bool:
    return len(scores) > 0 and float(max(scores)) >= threshold


def rerank_dataframe(
    query: str,
    candidates: pd.DataFrame,
    *,
    top_k: int,
    max_chunks_per_doc: int = 2,
    model: CrossEncoder | None = None,
) -> pd.DataFrame:
    if candidates.empty or top_k <= 0:
        return candidates.iloc[0:0].copy()
    results = candidates.copy()
    results["rerank_score"] = score_passages(
        query,
        results["chunk_text"].astype(str).tolist(),
        model=model,
    )
    results = results.sort_values("rerank_score", ascending=False, kind="stable")
    if max_chunks_per_doc > 0:
        results["_doc_rank"] = results.groupby("doc_id").cumcount()
        results = results[results["_doc_rank"] < max_chunks_per_doc].drop(columns="_doc_rank")
    return results.head(top_k).copy()
