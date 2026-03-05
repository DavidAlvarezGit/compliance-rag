# Simple A/B Evaluation

This folder gives a beginner-friendly benchmark:

- `A` = your RAG pipeline (`src.answer.answer_question`)
- `B` = normal chat model without retrieval context

## 1) Prepare questions

Edit `eval/questions.csv` and add more rows.

Required columns:

- `id`: integer
- `question`: user question
- `reference_answer`: short expected answer
- `is_answerable`: `1` if answer exists in your corpus, `0` if it should refuse

## 2) Run A/B generation

```powershell
poetry run python eval/run_ab.py
```

This creates `eval/results.csv`.

## 3) Score results

```powershell
poetry run python eval/score_ab.py
```

This creates `eval/results_scored.csv` and prints a summary.

## 4) Build shareable markdown report

```powershell
poetry run python eval/make_report.py
```

This creates `eval/report.md`.

## Metrics (simple)

- Keyword recall vs reference answer (proxy for correctness)
- Refusal accuracy on unanswerable questions
- Citation presence in RAG answers
- Latency comparison

## Important note

These are simple, understandable metrics.
For production-grade claims, also add human review and confidence intervals.
