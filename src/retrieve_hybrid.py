from __future__ import annotations

import pandas as pd

try:
    from .rag import RetrievalConfig, load_resources, retrieve_candidates
except ImportError:
    from rag import RetrievalConfig, load_resources, retrieve_candidates


def hybrid_search(query, top_k=5):
    resources = load_resources()
    results = retrieve_candidates(
        query,
        chunks_df=resources.chunks_df,
        embedder=resources.embedder,
        index=resources.index,
        config=RetrievalConfig(top_k=top_k),
    )
    return pd.DataFrame(
        [
            {
                "doc_id": result.doc_id,
                "page_start": result.page_start,
                "page_end": result.page_end,
                "topic": result.topic,
                "issue": result.issue,
                "language": result.language,
                "chunk_text": result.chunk_text,
                "bm25_score": result.bm25,
                "vector_score": result.vec,
                "hybrid_score": result.hybrid,
            }
            for result in results
        ]
    )
