# Banking Regulation Compliance Assistant

**Bilingual RAG system for grounded search and question answering over Basel and FINMA regulation.**

**Live demo:** https://compliance-rag.streamlit.app/

This project implements a retrieval-augmented generation pipeline for regulatory research. It supports questions in English and French over a curated 22-document corpus and returns answers with document- and page-level citations.

The system combines hybrid retrieval, multilingual reranking, grounded generation, and automated evidence verification. Unsupported claims are removed before display; if the available evidence is insufficient, the system refuses the query.

## Results

### Answer quality

| Metric                          |    Result |
| ------------------------------- | --------: |
| Correctness                     | **85.0%** |
| Completeness                    | **85.0%** |
| Retrieval Recall@8              | **90.0%** |
| Refusal accuracy                | **85.0%** |
| Factual-claim citation coverage |  **100%** |
| Citation provenance accuracy    | **99.2%** |
| Semantic evidence-support rate  | **98.3%** |
| Fully verified answer rate      | **80.0%** |

On a 20-question benchmark, the grounded system achieved **85% correctness and completeness**, compared with **75%** for the same language model without retrieval.

The main improvement comes from safer handling of unsupported questions. The stricter pipeline also introduces additional latency and can refuse partially supported answers rather than expose unsupported claims.

### Retrieval

| Metric                       | Hybrid | Hybrid + reranker | Change |
| ---------------------------- | -----: | ----------------: | -----: |
| Recall@5                     |  0.956 |             0.911 | -0.044 |
| Precision@5                  |  0.269 |             0.256 | -0.013 |
| Mean reciprocal rank         |  0.712 |         **0.794** | +0.082 |
| nDCG@5                       |  0.759 |         **0.805** | +0.046 |
| Unsupported-query abstention |    50% |           **80%** | +30 pp |

The reranker improves ranking quality and rejection of unsupported queries, with a small reduction in top-5 recall.

Warm-model retrieval latency increased from approximately **0.068 s to 3.425 s** on the benchmark machine. The application bounds the reranking candidate pool to limit unnecessary CPU work.

## Architecture

```text
Regulatory PDFs
       │
       ▼
PDF extraction
       │
       ▼
Paragraph-aware chunking
       │
       ├───────────────┐
       ▼               ▼
     BM25          Vector index
       │               │
       └───────┬───────┘
               ▼
        Hybrid retrieval
               │
               ▼
 Multilingual cross-encoder
           reranking
               │
               ▼
        Grounded LLM
         generation
               │
               ▼
   Citation and evidence
          verification
               │
        ┌──────┴──────┐
        ▼             ▼
 Verified answer    Refusal
```

## Retrieval Pipeline

The retrieval pipeline combines complementary search methods:

* **BM25** for exact regulatory terminology;
* **multilingual embeddings** for semantic retrieval across English and French;
* **FAISS** for vector search;
* a **cross-encoder reranker** that evaluates the question and candidate passage jointly;
* relevance and temporal checks for weak evidence and unsupported document years;
* source-diversity constraints to prevent one document from occupying every evidence position.

The interactive application reranks a bounded set of retrieved candidates before passing evidence to the generation model.

## Grounded Generation

The language model receives only the selected regulatory passages.

Answers are generated as concise factual claims with explicit document and page references, for example:

```text
(Source: governance-2017-corporate-governance pp.4-5)
```

Before an answer is displayed, the verification layer checks that:

1. factual claims contain citations;
2. the cited document and page range were present in the supplied context;
3. the cited evidence supports the claim;
4. the response addresses the question asked;
5. material constraints such as dates, jurisdictions, entities, products, and conditions have not been omitted.

If individual claims fail verification, they are removed. If the remaining supported content no longer provides a useful answer, the system returns an insufficient-evidence response.

Unchecked drafts are never streamed to the interface.

## Document Processing

`data/metadata/docs.csv` contains the document register used by the ingestion pipeline.

PDF pages are extracted, cleaned, and split into overlapping paragraph-level chunks. Each chunk retains stable document identifiers and page ranges so retrieved evidence can be traced back to its source.

The indexing pipeline creates:

* a BM25 keyword index;
* a FAISS vector index;
* associated metadata;
* an integrity manifest.

The integrity checks prevent the application from silently loading artifacts that no longer match the configured corpus, metadata, embedding dimension, or embedding model.

## Evaluation

### Retrieval benchmark

The retrieval benchmark contains **40 curated questions**:

* 30 answerable;
* 10 unsupported.

The labels are review-ready but have not been independently approved by a banking-regulation subject-matter expert.

The benchmark measures retrieval quality using Recall@5, Precision@5, mean reciprocal rank, nDCG@5, and unsupported-query abstention.

Evaluation outputs are available in:

```text
eval/retrieval_report.md
eval/retrieval_summary.json
eval/README.md
```

### Answer benchmark

The answer benchmark contains **20 separate questions** and compares:

```text
Grounded RAG pipeline
vs.
Same language model without retrieval
```

Unsupported questions use deterministic refusal scoring. Answerable questions use a structured language-model judge.

Mean end-to-end latency was:

| System                | Mean latency |
| --------------------- | -----------: |
| Grounded RAG          |     13.169 s |
| No-retrieval baseline |      5.186 s |

Evaluation outputs are available in:

```text
eval/report.md
eval/answer_summary.json
```

## Engineering

The repository separates ingestion, retrieval, reranking, generation, verification, evaluation, and application logic.

It includes:

* modular pipeline components;
* cached model and artifact loading;
* document and index integrity validation;
* deterministic and model-based grounding checks;
* English, French, paraphrase, multi-document, and unsupported-query test cases;
* versioned evaluation thresholds;
* continuous-integration quality gates;
* machine-readable benchmark outputs;
* **19 automated tests** covering ingestion, retrieval, ranking, artifacts, citations, question relevance, claim filtering, and refusal behavior.

## Project Structure

```text
app/
└── streamlit_app.py          interactive application

data/
├── metadata/
│   └── docs.csv              regulatory document register
├── processed/                extracted pages and chunks
└── artifacts/                indexes, metadata and integrity manifest

src/
├── parse_pdf.py              PDF extraction
├── chunk.py                  paragraph-aware chunking
├── index_embeddings.py       embedding and vector-index construction
├── retrieve.py               keyword-search baseline
├── retrieve_vector.py        vector-search baseline
├── retrieve_hybrid.py        hybrid retrieval pipeline
├── rerank.py                 multilingual cross-encoder reranking
├── answer.py                 generation and orchestration
├── citations.py              citation and evidence verification
└── artifacts.py              index and corpus integrity checks

eval/                         datasets, metrics, thresholds and reports
tests/                        automated tests
.github/workflows/ci.yml      CI and retrieval quality gate
```

## Running Locally

### Requirements

* Python 3.12+
* Poetry
* an `OPENAI_API_KEY` in `.env`

### Install and launch

```powershell
poetry install
poetry run python -m streamlit run app/streamlit_app.py
```

Example queries:

```text
What governance responsibilities does the board have for internal controls?

What does the operational resilience framework require for incident management?

Quelles obligations découlent de l'ordonnance sur les liquidités ?

What does the current corpus say about climate and nature-related financial risk governance?
```

## Configuration

The main optional environment variables are:

```text
OPENAI_MODEL
OPENAI_VERIFIER_MODEL
OPENAI_EVAL_MODEL
RERANKER_MODEL
RERANK_CANDIDATE_K
MIN_RERANK_SCORE
VERIFIED_STREAM_DELAY_SECONDS
```

Defaults:

```text
OPENAI_MODEL=gpt-4o-mini

OPENAI_VERIFIER_MODEL=<OPENAI_MODEL>

OPENAI_EVAL_MODEL=<OPENAI_MODEL>

RERANK_CANDIDATE_K=24

MIN_RERANK_SCORE=0.0

VERIFIED_STREAM_DELAY_SECONDS=0.018
```

The evaluation pipeline retains its benchmarked 40-candidate reranking configuration.

## Rebuilding the Corpus

```powershell
poetry run python src/metadata.py
poetry run python src/parse_pdf.py
poetry run python src/chunk.py
poetry run python src/index_embeddings.py
```

## Tests and Evaluation

Run the automated tests and retrieval quality gate:

```powershell
poetry run pytest -q
poetry run python -m eval.run_retrieval_benchmark --check
```

Run the answer benchmark:

```powershell
poetry run python -m eval.run_ab
poetry run python -m eval.score_ab --check
poetry run python -m eval.make_report
```

The answer benchmark sends its evaluation questions, reference answers, and retrieved regulatory excerpts to the configured OpenAI API.

## Limitations

The current system is intended for research and decision support rather than production regulatory decision-making.

A production deployment would require additional work around:

* independent subject-matter review of evaluation labels;
* larger adversarial and multilingual evaluation sets;
* regulation-version testing;
* authentication and authorization;
* audit logging and evidence retention;
* monitoring, tracing, cost reporting, and service-level objectives;
* lower-latency reranker serving;
* calibrated confidence estimates;
* more flexible recovery from partially supported answers;
* deployment beyond a single Streamlit process.

Final regulatory conclusions should still be reviewed by qualified professionals.
