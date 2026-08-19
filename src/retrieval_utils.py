from __future__ import annotations

import re

import numpy as np
import pandas as pd


def filter_candidates(
    chunks_df: pd.DataFrame,
    allowed_topics: set[str],
    allowed_languages: set[str],
) -> pd.DataFrame:
    candidates = chunks_df.copy()
    candidates["_global_idx"] = candidates.index.astype(int)
    if allowed_topics:
        candidates = candidates[candidates["topic"].isin(allowed_topics)]
    if allowed_languages and "language" in candidates.columns:
        candidates = candidates[candidates["language"].isin(allowed_languages)]
    return candidates.reset_index(drop=True)


def map_vector_results(
    raw_indices: np.ndarray,
    raw_scores: np.ndarray,
    candidate_df: pd.DataFrame,
    limit: int,
) -> list[tuple[int, float]]:
    global_to_local = {
        int(global_idx): local_idx
        for local_idx, global_idx in enumerate(candidate_df["_global_idx"].tolist())
    }
    pairs: list[tuple[int, float]] = []
    for raw_idx, raw_score in zip(raw_indices.astype(int), raw_scores.astype(float)):
        if raw_idx in global_to_local:
            pairs.append((global_to_local[raw_idx], float(raw_score)))
            if len(pairs) >= limit:
                break
    return pairs


def query_references_unknown_year(query: str, chunks_df: pd.DataFrame) -> bool:
    requested = {int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", query)}
    if not requested or "year" not in chunks_df.columns:
        return False
    available = {int(value) for value in chunks_df["year"].dropna().tolist()}
    return not requested.issubset(available)
