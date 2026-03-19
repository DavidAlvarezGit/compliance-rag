from __future__ import annotations

import unittest

import pandas as pd

from src.chunk import build_chunks_for_doc


class ChunkingTests(unittest.TestCase):
    def test_build_chunks_preserves_page_span_and_min_length(self):
        paragraph = "Sentence one. Sentence two. Sentence three. " * 40
        group = pd.DataFrame(
            [
                {
                    "doc_id": "doc-1",
                    "doc_type": "REG_BANK",
                    "topic": "other",
                    "year": 2024,
                    "issue": "",
                    "language": "EN",
                    "page": 1,
                    "text": paragraph,
                },
                {
                    "doc_id": "doc-1",
                    "doc_type": "REG_BANK",
                    "topic": "other",
                    "year": 2024,
                    "issue": "",
                    "language": "EN",
                    "page": 2,
                    "text": paragraph,
                },
            ]
        )
        chunks = build_chunks_for_doc(group)
        self.assertTrue(chunks)
        self.assertGreaterEqual(chunks[0]["page_start"], 1)
        self.assertGreaterEqual(chunks[0]["page_end"], chunks[0]["page_start"])
        self.assertGreaterEqual(len(chunks[0]["chunk_text"]), 600)


if __name__ == "__main__":
    unittest.main()
