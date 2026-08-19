import numpy as np
import pandas as pd

from src.rerank import has_relevant_passage, rerank_dataframe


class FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):
        del show_progress_bar
        return np.array([1.0 if "best" in passage else 0.0 for _, passage in pairs])


def test_reranker_reorders_and_limits_chunks_per_document() -> None:
    candidates = pd.DataFrame(
        {
            "doc_id": ["a", "a", "b"],
            "chunk_text": ["weak", "best", "also useful"],
            "hybrid_score": [0.9, 0.2, 0.8],
        }
    )

    results = rerank_dataframe(
        "query", candidates, top_k=3, max_chunks_per_doc=1, model=FakeCrossEncoder()
    )

    assert results["chunk_text"].tolist() == ["best", "also useful"]


def test_relevance_gate_requires_at_least_one_non_negative_score() -> None:
    assert has_relevant_passage([-1.2, 0.1])
    assert not has_relevant_passage([-1.2, -0.1])
    assert not has_relevant_passage([])
