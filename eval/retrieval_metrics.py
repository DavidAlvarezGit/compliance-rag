from __future__ import annotations

import math
from typing import Iterable


def parse_doc_ids(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def unique_ranked_doc_ids(values: Iterable[object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return float("nan")
    selected = retrieved[:k]
    if not selected:
        return 0.0
    return len(set(selected) & set(relevant)) / len(selected)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return float("nan")
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def score_ranking(retrieved: list[str], relevant: list[str], k: int = 5) -> dict[str, float]:
    return {
        f"recall_at_{k}": recall_at_k(retrieved, relevant, k),
        f"precision_at_{k}": precision_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        f"ndcg_at_{k}": ndcg_at_k(retrieved, relevant, k),
    }
