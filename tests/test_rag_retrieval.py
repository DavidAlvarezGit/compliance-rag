from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.rag import RetrievalConfig, retrieve_candidates


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class FakeIndex:
    metric_type = 0

    def search(self, query_vec, limit):
        del query_vec, limit
        return (
            np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
            np.array([[1, 2, 0]], dtype=np.int64),
        )


class RetrievalTests(unittest.TestCase):
    def test_filtered_vector_results_keep_global_row_alignment(self):
        chunks_df = pd.DataFrame(
            [
                {
                    "doc_id": "doc-en-1",
                    "topic": "other",
                    "language": "EN",
                    "page_start": 1,
                    "page_end": 1,
                    "issue": "",
                    "chunk_text": "english capital requirements and liquidity",
                },
                {
                    "doc_id": "doc-fr-1",
                    "topic": "other",
                    "language": "FR",
                    "page_start": 2,
                    "page_end": 2,
                    "issue": "",
                    "chunk_text": "français liquidité gouvernance risques",
                },
                {
                    "doc_id": "doc-en-2",
                    "topic": "capital_requirements_framework",
                    "language": "EN",
                    "page_start": 3,
                    "page_end": 3,
                    "issue": "",
                    "chunk_text": "english operational resilience governance",
                },
            ]
        )
        results = retrieve_candidates(
            "governance liquidity",
            chunks_df=chunks_df,
            embedder=FakeEmbedder(),
            index=FakeIndex(),
            config=RetrievalConfig(
                allowed_languages=frozenset({"FR"}),
                allowed_topics=frozenset({"other"}),
                top_k=3,
            ),
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].doc_id, "doc-fr-1")


if __name__ == "__main__":
    unittest.main()
