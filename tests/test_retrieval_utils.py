import numpy as np
import pandas as pd

from src.retrieval_utils import (
    filter_candidates,
    map_vector_results,
    query_references_unknown_year,
)


def test_filtered_faiss_indices_keep_global_row_identity() -> None:
    chunks = pd.DataFrame(
        {
            "doc_id": ["a", "b", "c"],
            "topic": ["excluded", "kept", "kept"],
            "language": ["EN", "EN", "FR"],
            "chunk_text": ["a", "b", "c"],
        }
    )
    candidates = filter_candidates(chunks, {"kept"}, set())

    pairs = map_vector_results(
        np.array([2, 0, 1]), np.array([0.9, 0.8, 0.7]), candidates, limit=2
    )

    assert candidates["_global_idx"].tolist() == [1, 2]
    assert pairs == [(1, 0.9), (0, 0.7)]


def test_unknown_explicit_year_is_outside_fixed_corpus() -> None:
    chunks = pd.DataFrame({"year": [2019, 2024, 2026]})

    assert query_references_unknown_year("What did Basel publish in 2035?", chunks)
    assert not query_references_unknown_year("What changed after 2024?", chunks)
    assert not query_references_unknown_year("What are the capital requirements?", chunks)
