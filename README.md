# Banking Regulation Compliance Assistant

**Live demo:** https://compliance-rag.streamlit.app/

## 1. Project Overview

This project is a bilingual assistant for searching and understanding Basel and FINMA banking regulation. It helps a user ask a question in English or French, find the relevant passages in a curated 22-document corpus, and receive an answer with inspectable document and page references.

The project is designed for situations where a fluent answer is not enough. Compliance officers, legal reviewers, and risk teams also need to know:

- which approved source supports the answer;
- where the supporting text appears;
- whether every factual claim is grounded;
- and when the available evidence is too weak to answer safely.

If the complete draft does not pass, the application removes every rejected claim and can show only the independently verified claims when they still answer the question. If no useful verified answer remains, it shows an insufficient-evidence response. This is **claim-level fail-closed behavior**.

The interface shows progress while it searches, reranks, drafts, and verifies. A successful response is then revealed gradually in a familiar chat-style animation. The unverified draft is never streamed to the user.

## 2. Problem Definition and Business Value

Banking regulation is long, fragmented, multilingual, and frequently difficult to search. Manual review is reliable but slow. A general-purpose chatbot is faster, but it can rely on outside knowledge, overlook an important condition, or produce a convincing answer without traceable evidence.

This assistant combines the speed of language models with a controlled evidence workflow. It is intended to make initial regulatory research faster and easier to review—not to replace professional legal or compliance judgment.

Compared with a normal chatbot, the system is designed to:

- answer from a fixed regulatory corpus rather than unrestricted model memory;
- cite the evidence behind every factual claim;
- preserve important dates, jurisdictions, entities, products, and conditions from the question;
- refuse questions that the corpus does not support;
- expose the retrieved passages and ranking scores for review;
- measure its quality through reproducible benchmarks and regression gates.

## 3. How the System Works

```text
Regulatory PDFs
      ↓
Clean page text and paragraph-aware chunks
      ↓
Keyword search + semantic vector search
      ↓
Multilingual relevance reranking
      ↓
Evidence-grounded answer draft
      ↓
Citation, page, claim-support, and question-relevance checks
      ↓
Verified answer — or an explicit refusal
```

### Document preparation

`data/metadata/docs.csv` is the source-of-truth document register. The ingestion pipeline extracts PDF pages, cleans the text, and builds overlapping paragraph-level chunks that retain stable document IDs and page ranges.

The system creates both a keyword index and a semantic FAISS vector index. An integrity manifest prevents the application from silently using an index that no longer matches the corpus, metadata, vector dimension, or embedding model.

### Retrieval and reranking

The application combines:

- **BM25 keyword search**, useful for exact regulatory terminology;
- **multilingual embeddings**, useful for paraphrases and bilingual questions;
- **cross-encoder reranking**, which reads the question and each candidate passage together for more accurate ordering;
- **relevance and temporal gates**, which reject weak evidence and unsupported document years;
- **source diversity limits**, which prevent one document from occupying every evidence position.

The interactive application reranks the best 40 candidates, matching the benchmarked pipeline and avoiding the previous cost of scoring as many as 80 passages.

### Answer generation and verification

The answer model receives only the selected source passages. It must write concise bullet points containing one factual claim and an exact citation such as:

```text
(Source: governance-2017-corporate-governance pp.4-5)
```

Before an answer is displayed, the verifier checks:

1. every factual claim has a citation;
2. the cited document and page range were actually supplied;
3. the cited excerpt directly supports the claim;
4. the answer addresses the complete question;
5. no material date, location, jurisdiction, entity, product, or condition was silently dropped.

This final check prevents a dangerous failure mode in which an answer is well cited but answers a broader, easier question than the one the user actually asked.

For safety, progressive display begins only after verification. The application never exposes the unchecked draft: it displays either the fully verified answer, a filtered answer containing only supported claims, or an immediate refusal when no relevant supported answer remains.

## 4. Engineering Highlights

The repository demonstrates more than a working interface. It includes:

- modular ingestion, retrieval, reranking, generation, and verification components;
- cached model and artifact loading;
- stage-by-stage progress and safe post-verification response streaming;
- explicit document, page, and index integrity contracts;
- deterministic and model-based grounding checks;
- English, French, paraphrase, multi-document, and unsupported test cases;
- versioned quality thresholds enforced in CI;
- machine-readable benchmark summaries and shareable reports;
- 18 automated tests covering ingestion, retrieval, ranking, artifacts, citations, question relevance, claim-level filtering, and refusal behavior;
- transparent reporting of both improvements and tradeoffs.

## 5. Measured Results

### Retrieval benchmark

The retrieval dataset contains 40 curated questions: 30 answerable and 10 unsupported. The labels are review-ready but have not been independently approved by a banking-regulation subject-matter expert.

| Metric | Hybrid baseline | Hybrid + reranker | Change |
|---|---:|---:|---:|
| Recall@5 | 0.956 | 0.911 | -0.044 |
| Precision@5 | 0.269 | 0.256 | -0.013 |
| Mean reciprocal rank | 0.712 | 0.794 | +0.082 |
| nDCG@5 ranking quality | 0.759 | 0.805 | +0.046 |
| Unsupported-question abstention | 50% | 80% | +30 points |

The reranker improves first-result quality, overall ordering, and unsupported-question rejection, with a small recall tradeoff. Warm-model retrieval latency increased from approximately `0.068s` to `3.425s` on the benchmark machine. The application now caps reranking at the same 40-candidate benchmark size to avoid unnecessary CPU work.

See the [retrieval report](eval/retrieval_report.md), [machine-readable summary](eval/retrieval_summary.json), and [evaluation methodology](eval/README.md).

### Answer-quality benchmark

The answer benchmark contains 20 separate questions and compares the grounded system with the same language model operating without retrieval.

| Metric | RAG assistant | No-retrieval baseline |
|---|---:|---:|
| Correctness | 0.850 | 0.750 |
| Completeness | 0.850 | 0.750 |
| Mean end-to-end latency | 13.169s | 5.186s |

Additional RAG measurements:

- retrieval recall@8: **90%**;
- refusal accuracy: **85%**;
- factual-claim citation coverage: **100%**;
- citation provenance accuracy: **99.2%**;
- semantic evidence-support rate: **98.3%**;
- fully verified answer rate: **80%**.

The overall correctness advantage comes mainly from safer behavior on unsupported questions. Successful answers usually match the baseline on answerable questions, while the strict system sometimes refuses a partially supported draft rather than expose it. This safety choice improves trustworthiness but reduces coverage and adds latency.

Answerable questions use a structured language-model judge. Unsupported questions use deterministic refusal scoring because repeated testing showed that a model judge could score identical refusals inconsistently. See the [answer-quality report](eval/report.md) and [machine-readable summary](eval/answer_summary.json).

## 6. Run the Project Locally

### Prerequisites

- Python 3.12+
- Poetry
- an `OPENAI_API_KEY` in `.env`

### Install and start the application

```powershell
poetry install
poetry run python -m streamlit run app/streamlit_app.py
```

Example questions:

```text
What governance responsibilities does the board have for internal controls?
What does the operational resilience framework require for incident management?
Quelles obligations découlent de l'ordonnance sur les liquidités ?
What does the current corpus say about climate and nature-related financial risk governance?
```

### Optional configuration

- `OPENAI_MODEL`: answer model; default `gpt-4o-mini`
- `OPENAI_VERIFIER_MODEL`: semantic verifier; defaults to the answer model
- `OPENAI_EVAL_MODEL`: benchmark judge; defaults to the answer model
- `RERANKER_MODEL`: cross-encoder model override
- `RERANK_CANDIDATE_K`: interactive reranking pool; default `40`
- `MIN_RERANK_SCORE`: top-passage relevance threshold; default `0.0`
- `VERIFIED_STREAM_DELAY_SECONDS`: delay between displayed words after verification; default `0.018`

### Rebuild the corpus and indexes

```powershell
poetry run python src/metadata.py
poetry run python src/parse_pdf.py
poetry run python src/chunk.py
poetry run python src/index_embeddings.py
```

### Run tests and evaluation gates

```powershell
poetry run pytest -q
poetry run python -m eval.run_retrieval_benchmark --check
```

The answer benchmark sends its evaluation questions, reference answers, and retrieved regulatory excerpts to the configured OpenAI API:

```powershell
poetry run python -m eval.run_ab
poetry run python -m eval.score_ab --check
poetry run python -m eval.make_report
```

## 7. Project Structure

```text
app/streamlit_app.py             interactive application and inspection controls
data/metadata/docs.csv           curated regulatory document register
data/processed/                  extracted pages and retrieval chunks
data/artifacts/                  FAISS index, metadata, and integrity manifest
src/parse_pdf.py                 PDF text extraction
src/chunk.py                     paragraph-aware chunking
src/index_embeddings.py          embedding and index construction
src/retrieve.py                  standalone keyword-search baseline
src/retrieve_vector.py           standalone vector-search baseline
src/retrieve_hybrid.py           production hybrid retrieval pipeline
src/rerank.py                    multilingual cross-encoder reranking
src/answer.py                    grounded generation and fail-closed orchestration
src/citations.py                 citation and semantic verification
src/artifacts.py                 index and corpus integrity validation
eval/                            datasets, metrics, thresholds, results, and reports
tests/                           automated regression tests
.github/workflows/ci.yml         continuous integration and retrieval quality gate
```

## 8. Current Limitations and Next Steps

This is a strong engineering and portfolio project, not yet a regulated production service. The next phase should focus on:

- independent review of evaluation labels by compliance specialists;
- larger adversarial, multilingual, and regulation-version test sets;
- authentication, authorization, and user-level access controls;
- audit logs and evidence-retention policies;
- monitoring, tracing, cost reporting, and service-level objectives;
- faster reranker serving or a smaller latency-optimized model;
- calibrated answer confidence and less all-or-nothing claim recovery;
- deployment beyond a single Streamlit process.

The current application should be treated as a research and decision-support tool. Final regulatory conclusions still require review by qualified professionals.
