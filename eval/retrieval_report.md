# Retrieval Benchmark

## Dataset

- Questions: 40
- Answerable: 30
- Unanswerable: 10
- Status: curated and review-ready; not represented as expert/SME approved
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Latency: warm-model steady state

## Results

| Metric | Hybrid retrieval |
|---|---:|
| Recall@5 | 0.956 |
| Precision@5 | 0.269 |
| MRR | 0.712 |
| nDCG@5 | 0.759 |
| Unsupported-query abstention | 0.500 |

## Latency

- Mean / p95: 0.053s / 0.067s

Metrics are document-level. Multi-document questions receive full recall only when all labeled sources are retrieved.
Unanswerable questions measure whether the retrieval relevance gate returns no evidence.
