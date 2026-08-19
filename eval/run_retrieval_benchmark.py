from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from eval.retrieval_metrics import parse_doc_ids, score_ranking, unique_ranked_doc_ids
from src.rerank import RERANKER_MODEL
from src.retrieve_hybrid import EMBEDDING_MODEL, hybrid_search

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "eval"
QUESTIONS_PATH = EVAL_DIR / "retrieval_questions.csv"
RESULTS_PATH = EVAL_DIR / "retrieval_results.csv"
SUMMARY_PATH = EVAL_DIR / "retrieval_summary.json"
REPORT_PATH = EVAL_DIR / "retrieval_report.md"
THRESHOLDS_PATH = EVAL_DIR / "retrieval_thresholds.json"
K = 5


def _run(query: str, *, rerank: bool) -> tuple[list[str], float]:
    started = time.perf_counter()
    results = hybrid_search(
        query,
        top_k=10,
        max_chunks_per_doc=2,
        rerank=rerank,
        candidate_k=40,
    )
    latency = time.perf_counter() - started
    return unique_ranked_doc_ids(results["doc_id"].tolist()), latency


def _aggregate(results: pd.DataFrame, prefix: str) -> dict[str, float]:
    answerable = results[results["is_answerable"].astype(int) == 1]
    unanswerable = results[results["is_answerable"].astype(int) == 0]
    return {
        "recall_at_5": float(answerable[f"{prefix}_recall_at_5"].mean()),
        "precision_at_5": float(answerable[f"{prefix}_precision_at_5"].mean()),
        "mrr": float(answerable[f"{prefix}_mrr"].mean()),
        "ndcg_at_5": float(answerable[f"{prefix}_ndcg_at_5"].mean()),
        "abstention_accuracy": float(unanswerable[f"{prefix}_abstained"].mean()),
        "latency_mean_s": float(results[f"{prefix}_latency_s"].mean()),
        "latency_p95_s": float(results[f"{prefix}_latency_s"].quantile(0.95)),
    }


def _report(summary: dict[str, object]) -> str:
    baseline = summary["baseline"]
    reranked = summary["reranked"]
    lines = [
        "# Retrieval Benchmark",
        "",
        "## Dataset",
        "",
        f"- Questions: {summary['questions']}",
        f"- Answerable: {summary['answerable_questions']}",
        f"- Unanswerable: {summary['unanswerable_questions']}",
        "- Status: curated and review-ready; not represented as expert/SME approved",
        f"- Embedding model: `{summary['embedding_model']}`",
        f"- Reranker: `{summary['reranker_model']}`",
        f"- Latency: {summary['latency_mode']}",
        f"- Latency: {summary['latency_mode']}",
        "",
        "## Results",
        "",
        "| Metric | Hybrid baseline | Hybrid + reranker | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("recall_at_5", "precision_at_5", "mrr", "ndcg_at_5", "abstention_accuracy"):
        before = float(baseline[metric])
        after = float(reranked[metric])
        lines.append(f"| {metric} | {before:.3f} | {after:.3f} | {after - before:+.3f} |")
    lines.extend(
        [
            "",
            "## Latency",
            "",
            f"- Baseline mean / p95: {baseline['latency_mean_s']:.3f}s / {baseline['latency_p95_s']:.3f}s",
            f"- Reranked mean / p95: {reranked['latency_mean_s']:.3f}s / {reranked['latency_p95_s']:.3f}s",
            "",
            "Metrics are document-level. Multi-document questions receive full recall only when all labeled sources are retrieved.",
            "Unanswerable questions measure whether the retrieval relevance gate returns no evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _check(summary: dict[str, object]) -> None:
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    failures = []
    reranked = summary["reranked"]
    baseline = summary["baseline"]
    for metric, minimum in thresholds["minimums"].items():
        if float(reranked[metric]) < float(minimum):
            failures.append(f"{metric}={reranked[metric]:.3f} < {minimum:.3f}")
    for metric, allowed_drop in thresholds["maximum_drop_from_baseline"].items():
        delta = float(reranked[metric]) - float(baseline[metric])
        if delta < -float(allowed_drop):
            failures.append(f"{metric} delta={delta:+.3f} exceeds allowed drop {allowed_drop:.3f}")
    if failures:
        raise SystemExit("Retrieval regression gate failed: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail when configured thresholds regress.")
    args = parser.parse_args()

    questions = pd.read_csv(QUESTIONS_PATH)
    # Load models and touch both paths before measuring steady-state request latency.
    warmup_query = str(questions.iloc[0]["question"])
    hybrid_search(warmup_query, top_k=10, max_chunks_per_doc=2, rerank=False, candidate_k=40)
    hybrid_search(warmup_query, top_k=10, max_chunks_per_doc=2, rerank=True, candidate_k=40)
    rows = []
    for question in questions.itertuples(index=False):
        relevant = parse_doc_ids(question.relevant_doc_ids)
        baseline_docs, baseline_latency = _run(str(question.question), rerank=False)
        reranked_docs, reranked_latency = _run(str(question.question), rerank=True)
        row = question._asdict()
        row.update(
            {
                "baseline_doc_ids": "|".join(baseline_docs),
                "reranked_doc_ids": "|".join(reranked_docs),
                "baseline_latency_s": baseline_latency,
                "reranked_latency_s": reranked_latency,
                "baseline_abstained": int(not baseline_docs),
                "reranked_abstained": int(not reranked_docs),
            }
        )
        row.update({f"baseline_{key}": value for key, value in score_ranking(baseline_docs, relevant, K).items()})
        row.update({f"reranked_{key}": value for key, value in score_ranking(reranked_docs, relevant, K).items()})
        rows.append(row)
        print(f"Evaluated retrieval question {question.id}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    summary = {
        "questions": len(results),
        "answerable_questions": int(results["is_answerable"].sum()),
        "unanswerable_questions": int((results["is_answerable"] == 0).sum()),
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "latency_mode": "warm-model steady state",
        "baseline": _aggregate(results, "baseline"),
        "reranked": _aggregate(results, "reranked"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.check:
        _check(summary)


if __name__ == "__main__":
    main()
