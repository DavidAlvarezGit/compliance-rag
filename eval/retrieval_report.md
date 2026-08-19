# Retrieval Benchmark

## Dataset

- Questions: 40
- Answerable: 30
- Unanswerable: 10
- Status: curated and review-ready; not represented as expert/SME approved
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Latency: warm-model steady state
- Latency: warm-model steady state

## Results

| Metric | Hybrid baseline | Hybrid + reranker | Delta |
|---|---:|---:|---:|
| recall_at_5 | 0.956 | 0.911 | -0.044 |
| precision_at_5 | 0.269 | 0.256 | -0.013 |
| mrr | 0.712 | 0.794 | +0.082 |
| ndcg_at_5 | 0.759 | 0.805 | +0.046 |
| abstention_accuracy | 0.500 | 0.800 | +0.300 |

## Latency

- Baseline mean / p95: 0.068s / 0.096s
- Reranked mean / p95: 3.425s / 4.009s

Metrics are document-level. Multi-document questions receive full recall only when all labeled sources are retrieved.
Unanswerable questions measure whether the retrieval relevance gate returns no evidence.
