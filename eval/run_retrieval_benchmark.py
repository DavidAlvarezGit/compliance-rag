from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from eval.retrieval_metrics import parse_doc_ids, score_ranking, unique_ranked_doc_ids
from src.retrieve_hybrid import EMBEDDING_MODEL, hybrid_search

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "eval"
QUESTIONS_PATH = EVAL_DIR / "retrieval_questions.csv"
RESULTS_PATH = EVAL_DIR / "retrieval_results.csv"
SUMMARY_PATH = EVAL_DIR / "retrieval_summary.json"
REPORT_PATH = EVAL_DIR / "retrieval_report.md"
THRESHOLDS_PATH = EVAL_DIR / "retrieval_thresholds.json"
K = 5


def _run(query: str) -> tuple[list[str], float]:
    started = time.perf_counter()
    results = hybrid_search(query, top_k=10, max_chunks_per_doc=2)
    latency = time.perf_counter() - started
    return unique_ranked_doc_ids(results["doc_id"].tolist()), latency


def _aggregate(results: pd.DataFrame) -> dict[str, float]:
    answerable = results[results["is_answerable"].astype(int) == 1]
    unanswerable = results[results["is_answerable"].astype(int) == 0]
    return {
        "recall_at_5": float(answerable["recall_at_5"].mean()),
        "precision_at_5": float(answerable["precision_at_5"].mean()),
        "mrr": float(answerable["mrr"].mean()),
        "ndcg_at_5": float(answerable["ndcg_at_5"].mean()),
        "abstention_accuracy": float(unanswerable["abstained"].mean()),
        "latency_mean_s": float(results["latency_s"].mean()),
        "latency_p95_s": float(results["latency_s"].quantile(0.95)),
    }


def _report(summary: dict[str, object]) -> str:
    hybrid = summary["hybrid"]
    return "\n".join(
        [
            "# Retrieval Benchmark",
            "",
            "## Dataset",
            "",
            f"- Questions: {summary['questions']}",
            f"- Answerable: {summary['answerable_questions']}",
            f"- Unanswerable: {summary['unanswerable_questions']}",
            "- Status: curated and review-ready; not represented as expert/SME approved",
            f"- Embedding model: `{summary['embedding_model']}`",
            f"- Latency: {summary['latency_mode']}",
            "",
            "## Results",
            "",
            "| Metric | Hybrid retrieval |",
            "|---|---:|",
            f"| Recall@5 | {hybrid['recall_at_5']:.3f} |",
            f"| Precision@5 | {hybrid['precision_at_5']:.3f} |",
            f"| MRR | {hybrid['mrr']:.3f} |",
            f"| nDCG@5 | {hybrid['ndcg_at_5']:.3f} |",
            f"| Unsupported-query abstention | {hybrid['abstention_accuracy']:.3f} |",
            "",
            "## Latency",
            "",
            f"- Mean / p95: {hybrid['latency_mean_s']:.3f}s / {hybrid['latency_p95_s']:.3f}s",
            "",
            "Metrics are document-level. Multi-document questions receive full recall only when all labeled sources are retrieved.",
            "Unanswerable questions measure whether the retrieval relevance gate returns no evidence.",
            "",
        ]
    )


def _check(summary: dict[str, object]) -> None:
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    failures = []
    hybrid = summary["hybrid"]
    for metric, minimum in thresholds["minimums"].items():
        if float(hybrid[metric]) < float(minimum):
            failures.append(f"{metric}={hybrid[metric]:.3f} < {minimum:.3f}")
    if failures:
        raise SystemExit("Retrieval regression gate failed: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail when thresholds regress.")
    args = parser.parse_args()

    questions = pd.read_csv(QUESTIONS_PATH)
    warmup_query = str(questions.iloc[0]["question"])
    hybrid_search(warmup_query, top_k=10, max_chunks_per_doc=2)
    rows = []
    for question in questions.itertuples(index=False):
        relevant = parse_doc_ids(question.relevant_doc_ids)
        retrieved_docs, latency = _run(str(question.question))
        row = question._asdict()
        row.update(
            {
                "retrieved_doc_ids": "|".join(retrieved_docs),
                "latency_s": latency,
                "abstained": int(not retrieved_docs),
            }
        )
        row.update(score_ranking(retrieved_docs, relevant, K))
        rows.append(row)
        print(f"Evaluated retrieval question {question.id}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    summary = {
        "questions": len(results),
        "answerable_questions": int(results["is_answerable"].sum()),
        "unanswerable_questions": int((results["is_answerable"] == 0).sum()),
        "embedding_model": EMBEDDING_MODEL,
        "latency_mode": "warm-model steady state",
        "hybrid": _aggregate(results),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.check:
        _check(summary)


if __name__ == "__main__":
    main()
