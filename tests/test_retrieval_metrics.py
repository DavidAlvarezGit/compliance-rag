import math

from eval.retrieval_metrics import (
    ndcg_at_k,
    parse_doc_ids,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unique_ranked_doc_ids,
)


def test_document_ranking_metrics_support_multiple_relevant_sources() -> None:
    retrieved = ["wrong", "a", "b"]
    relevant = ["a", "b"]

    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert 0.0 < ndcg_at_k(retrieved, relevant, 3) < 1.0


def test_metric_helpers_handle_empty_labels_and_duplicate_chunks() -> None:
    assert parse_doc_ids("a|b") == ["a", "b"]
    assert unique_ranked_doc_ids(["a", "a", "b"]) == ["a", "b"]
    assert math.isnan(recall_at_k(["a"], [], 5))
