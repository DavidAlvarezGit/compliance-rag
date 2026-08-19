# Banking Regulation Compliance Assistant

**Live demo:** https://compliance-rag.streamlit.app/

## Overview

This project is a bilingual, evidence-grounded assistant for Basel and FINMA banking regulation. It retrieves from a fixed 22-document corpus, reranks candidate passages, generates an answer only from supplied evidence, and verifies every factual claim before showing it.

The central safety property is fail-closed behavior: if retrieval finds no sufficiently relevant evidence, a citation points outside the supplied document/page range, semantic support cannot be verified, or verification itself fails, the application returns an explicit insufficient-evidence response instead of the draft.

Unlike a conventional chatbot, the assistant does not treat fluent text as sufficient. A response must be relevant to the complete question, traceable to the supplied document pages, and semantically supported at claim level before it becomes visible.

## Architecture

```text
PDF registry -> page extraction -> paragraph chunks -> BM25 + FAISS
                                                        |
User query -> candidate fusion -> multilingual cross-encoder reranker
                                      |
                               grounded answer draft
                                      |
                     claim extraction + provenance checks
                                      |
                     semantic evidence-support verifier
                                      |
                         verified answer or refusal
```

- `data/metadata/docs.csv` is the source-of-truth document registry.
- Paragraph-aware chunks retain stable document IDs and page spans.
- BM25 and multilingual embeddings provide complementary candidate retrieval.
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reranks a wider candidate pool.
- Citations use `(Source: DOC_ID pp.X-Y)` and are checked against the exact context supplied to the model.
- The FAISS manifest validates model name, dimension, row count, metadata hash, and corpus hash at startup.

## Request lifecycle and safety checks

1. The query is checked for empty input and unsupported temporal references.
2. BM25 and FAISS independently retrieve keyword and semantic candidates.
3. Their normalized scores are fused into a wider candidate pool.
4. A multilingual cross-encoder reranks the query-passage pairs.
5. A relevance gate rejects the request when no passage clears the configured threshold.
6. The answer model receives only the selected excerpts and must produce one cited factual claim per bullet.
7. Deterministic validation confirms that every cited document and page range was actually supplied.
8. A semantic verifier checks claim support and whether the answer preserved every material entity, jurisdiction, location, date, product, and condition from the question.
9. Any failure replaces the draft with a same-language insufficient-evidence response.

This last relevance check prevents a particularly dangerous RAG failure: producing a well-cited answer about a related general topic while silently ignoring a qualifier in the original question.

## Measured retrieval results

The included benchmark contains 40 curated questions: 30 answerable and 10 unsupported, spanning English, French, paraphrases, and multi-document questions. Labels are review-ready but have not been independently approved by a banking-regulation SME.

| Metric | Hybrid baseline | Hybrid + reranker | Delta |
|---|---:|---:|---:|
| Recall@5 | 0.956 | 0.911 | -0.044 |
| Precision@5 | 0.269 | 0.256 | -0.013 |
| MRR | 0.712 | 0.794 | +0.082 |
| nDCG@5 | 0.759 | 0.805 | +0.046 |
| Unsupported-query abstention | 0.500 | 0.800 | +0.300 |

Reranking materially improves first-result quality, ranking quality, and unsupported-query abstention, with a small recall tradeoff. Warm-model mean latency changes from 0.068s to 3.425s on the benchmark machine; this is an explicit production optimization target, not hidden from the result. See the [retrieval report](eval/retrieval_report.md), [machine-readable summary](eval/retrieval_summary.json), and [evaluation methodology](eval/README.md).

The separate 20-question answer benchmark uses an LLM judge for answerable questions and deterministic refusal scoring for unsupported questions.

| Answer metric | RAG | No-retrieval baseline |
|---|---:|---:|
| Correctness | 0.850 | 0.750 |
| Completeness | 0.850 | 0.750 |
| Mean latency | 13.169s | 5.186s |

RAG retrieval recall@8 is 90%; refusal accuracy is 85%; claim citation coverage is 100%; citation provenance accuracy is 99.2%; semantic citation support is 98.3%; and 80% of answerable responses pass the complete fail-closed verifier. The overall correctness advantage comes from safe abstention on unsupported questions. On answerable questions the baseline generally ties successful RAG answers, while three RAG drafts fail closed rather than expose a partially unsupported response.

See the [answer-quality report](eval/report.md) and [machine-readable summary](eval/answer_summary.json). Answerable questions use a structured LLM judge; unsupported questions use deterministic scoring because repeated evaluation showed that an LLM judge scored identical refusals inconsistently.

## Run locally

Prerequisites: Python 3.12+, Poetry, and `OPENAI_API_KEY` in `.env`.

```powershell
poetry install
poetry run python -m streamlit run app/streamlit_app.py
```

Example question:

```text
What governance responsibilities does the board have for internal controls?
```

Optional configuration:

- `OPENAI_MODEL`: answer model (default `gpt-4o-mini`)
- `OPENAI_VERIFIER_MODEL`: semantic citation verifier (defaults to the answer model)
- `OPENAI_EVAL_MODEL`: answer-quality judge (defaults to the answer model)
- `RERANKER_MODEL`: sentence-transformers cross-encoder override
- `MIN_RERANK_SCORE`: top-passage relevance gate (default `0.0`)

Rebuild all corpus artifacts:

```powershell
poetry run python src/metadata.py
poetry run python src/parse_pdf.py
poetry run python src/chunk.py
poetry run python src/index_embeddings.py
```

Run tests and the local retrieval quality gate:

```powershell
poetry run pytest -q
poetry run python -m eval.run_retrieval_benchmark --check
```

Run the OpenAI-backed answer benchmark only when sending its questions and retrieved excerpts to the configured API is acceptable:

```powershell
poetry run python -m eval.run_ab
poetry run python -m eval.score_ab --check
poetry run python -m eval.make_report
```

## Repository map

```text
app/streamlit_app.py             Streamlit UI and inspection controls
src/retrieve_hybrid.py           BM25/FAISS fusion, gates, and candidate selection
src/retrieve.py                  standalone BM25 retrieval baseline
src/retrieve_vector.py           standalone FAISS retrieval baseline
src/rerank.py                    multilingual cross-encoder reranking
src/answer.py                    grounded generation and fail-closed orchestration
src/citations.py                 claim-level citation and semantic verification
src/artifacts.py                 artifact manifest validation
eval/retrieval_questions.csv     40-question retrieval set
eval/run_retrieval_benchmark.py  before/after benchmark and regression gate
eval/answer_questions.csv        20-question answer-quality set
eval/run_ab.py                   RAG-versus-baseline generation and judging
tests/                           ingestion, retrieval, ranking, grounding, and eval tests
.github/workflows/ci.yml         unit tests and retrieval quality gate
```

The current suite contains 16 tests covering ingestion, artifact integrity, retrieval utilities, ranking metrics, reranker behavior, citation provenance, semantic question relevance, and fail-closed refusal behavior.

## Production-readiness boundaries

This is a strong portfolio/engineering prototype, not yet a regulated production service. The next production phase should focus on independently reviewed labels, adversarial and temporal-version tests, calibrated abstention, reranker latency reduction or serving, authentication and authorization, audit/event logging, observability and SLOs, data-retention controls, and a deployment architecture beyond a single Streamlit process.
