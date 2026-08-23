# Banking Regulation Compliance Assistant

Ask questions about Swiss banking regulation in English or French and receive clear, verified answers with direct references to the relevant document and page. Switzerland is treated as the default jurisdiction unless another jurisdiction is specified.

**Live demo:** https://compliance-rag.streamlit.app/

## Why this project exists

Regulatory documents are long and difficult to search. A normal language model can answer quickly, but it may use outside knowledge or state something that the source does not support.

This project:

- searches through a collection of 22 regulatory documents;
- combines exact-term and meaning-based search;
- answers only from the passages it finds;
- cites the document and page for each claim;
- removes claims that the evidence does not support;
- refuses to answer when the available material is not enough.

It is however not a replacement for legal or compliance review.

## How it works

1. PDF pages are extracted and split into passages while keeping the document ID and page range.
2. BM25 finds exact regulatory terms.
3. Multilingual embeddings find passages with similar meaning in English and French.
4. The two search scores are combined and the strongest passages are sent to the language model.
5. The model writes short answers with document and page citations.
6. A final check confirms that each claim is cited, supported by the cited text, and relevant to the question.

If one claim fails, it is removed. If the remaining claims no longer answer the question, the application returns an insufficient-evidence message.

## Results

### Answers

The answer benchmark contains 50 questions: 36 answerable and 14 unsupported. It compares this project with the same model answering without retrieved documents.

| Measure | This project | Model without retrieval |
|---|---:|---:|
| Overall correctness | 95.0% | 72.0% |
| Overall completeness | 95.5% | 72.0% |
| Correctness on answerable questions | 98.6% | 100% |
| Unsupported questions refused | 85.7% | 0% |
| Mean response time | 7.19 seconds | 4.87 seconds |

The model without retrieval scored slightly higher on the answerable questions. This project scored higher overall because it refused 12 of 14 unsupported questions; the model without retrieval refused none. The two missed refusals and one partly answered supported question remain in the published results.

For answerable questions, the system found 93.5% of the expected documents in its first eight results. All returned claims had citations, valid source pages, and evidence support in this run.

### Search

The search benchmark contains 40 questions: 30 answerable and 10 unsupported.

| Measure | Result |
|---|---:|
| Recall@5 | 0.956 |
| Precision@5 | 0.269 |
| Mean reciprocal rank | 0.712 |
| nDCG@5 | 0.759 |
| Unsupported questions rejected during search | 50% |
| Mean warm search time | about 0.07 seconds |

Unsupported questions that pass the search stage still go through evidence validation before an answer can be shown.

## Example questions

```text
What governance responsibilities does the board have for internal controls?

What does the operational resilience framework require for incident management?

What does the current corpus say about climate and nature-related financial risk governance?
```

## Run locally

Requirements:

- Python 3.12+
- Poetry
- an `OPENAI_API_KEY` in `.env`

```powershell
poetry install
poetry run python -m streamlit run app/streamlit_app.py
```

Optional environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_MODEL` | Answer model | `gpt-4o-mini` |
| `OPENAI_VERIFIER_MODEL` | Evidence-checking model | Same as `OPENAI_MODEL` |
| `OPENAI_EVAL_MODEL` | Answer benchmark judge | Same as `OPENAI_MODEL` |
| `HYBRID_CANDIDATE_K` | Search candidate pool | `40` |
| `MIN_VECTOR_SIMILARITY` | Minimum semantic match | `0.25` |

## Rebuild the document index

```powershell
poetry run python src/metadata.py
poetry run python src/parse_pdf.py
poetry run python src/chunk.py
poetry run python src/index_embeddings.py
```

## Tests and evaluation

Run the 18 automated tests and the local search-quality gate:

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

The answer benchmark sends its questions, reference answers, and retrieved regulatory passages to the configured OpenAI API.

Detailed results are available in:

- [search benchmark](eval/retrieval_report.md)
- [answer benchmark](eval/report.md)
- [evaluation notes](eval/README.md)

## Project structure

```text
app/                   Streamlit application
src/                   document processing, search, answers, and validation
data/metadata/         document register
data/processed/        extracted pages and passages
data/artifacts/        search index and integrity metadata
eval/                  benchmark datasets, thresholds, and reports
tests/                 automated tests
.github/workflows/     continuous integration
```

## Current limits

Before production use, the project would need:

- independent review by regulatory specialists;
- a larger and more adversarial test set;
- authentication and access control;
- monitoring for quality, latency, and cost;
- a deployment designed for multiple users.

