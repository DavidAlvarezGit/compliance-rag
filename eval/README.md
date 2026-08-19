# Evaluation

The evaluation suite separates retrieval quality from generated-answer quality so a failure can be attributed to the correct pipeline stage.

## Retrieval benchmark (local, deterministic)

`retrieval_questions.csv` contains 40 document-labeled questions: English, French, paraphrase, multi-document, and unsupported cases. Labels are curated and review-ready, but are not represented as expert/SME approved.

Run the benchmark and its regression gate:

```powershell
poetry run python -m eval.run_retrieval_benchmark --check
```

It compares hybrid BM25 + FAISS retrieval with and without cross-encoder reranking. Outputs:

- `retrieval_results.csv`: per-question rankings, metrics, and latency
- `retrieval_summary.json`: machine-readable aggregate results
- `retrieval_report.md`: shareable before/after report
- `retrieval_thresholds.json`: versioned CI acceptance criteria

Metrics are document-level recall@5, precision@5, MRR, nDCG@5, unsupported-question abstention accuracy, and warm-model latency. The CI workflow runs this gate without an OpenAI key.

## Answer-quality benchmark (external API)

`answer_questions.csv` contains 20 curated answer-quality questions with reference answers and expected source documents. Running this benchmark sends the questions and retrieved corpus excerpts to the configured OpenAI API.

```powershell
poetry run python -m eval.run_ab
poetry run python -m eval.score_ab --check
poetry run python -m eval.make_report
```

The benchmark compares the reranked, citation-verified RAG pipeline with the same chat model without retrieval. It measures:

- LLM-judged factual correctness and completeness for answerable questions (0-4 rubric)
- deterministic unsupported-question scoring (canonical refusal = 4; substantive answer = 0)
- document recall@8
- refusal accuracy
- claim-level citation coverage
- deterministic citation provenance accuracy
- semantic evidence-support rate
- verified-answer rate and latency

`answer_thresholds.json` contains the predeclared answer-quality gate. Generated outputs are `results.csv`, `results_scored.csv`, `answer_summary.json`, and `report.md`.

## Interpretation limits

- The datasets are curated, not independently SME-approved.
- LLM-as-judge scores are useful regression signals, not substitutes for human review.
- A production release should add blinded compliance-expert review, inter-rater agreement, confidence intervals, and a larger adversarial set.
