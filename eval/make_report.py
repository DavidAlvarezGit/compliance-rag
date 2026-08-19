"""
Build a simple markdown report from eval/results_scored.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
SCORED_PATH = BASE_DIR / "eval" / "results_scored.csv"
REPORT_PATH = BASE_DIR / "eval" / "report.md"


def short_text(text: str, max_len: int = 280) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def build_example_block(df: pd.DataFrame, title: str, limit: int = 3) -> str:
    lines = [f"## {title}", ""]
    if df.empty:
        lines.append("No examples.")
        lines.append("")
        return "\n".join(lines)

    for row in df.head(limit).itertuples(index=False):
        lines.append(f"### Q{int(row.id)}")
        lines.append(f"**Question:** {row.question}")
        lines.append(f"**Reference:** {row.reference_answer}")
        lines.append(f"**RAG:** {short_text(str(row.rag_answer))}")
        lines.append(f"**Baseline:** {short_text(str(row.baseline_answer))}")
        lines.append(
            f"**Judged correctness:** RAG={int(row.rag_correctness)}/4, "
            f"Baseline={int(row.baseline_correctness)}/4"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not SCORED_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {SCORED_PATH}. Run eval/run_ab.py and eval/score_ab.py first."
        )

    df = pd.read_csv(SCORED_PATH)
    n = len(df)
    if n == 0:
        raise ValueError("results_scored.csv is empty.")

    rag_correctness = df["rag_correctness_normalized"].mean()
    base_correctness = df["baseline_correctness_normalized"].mean()
    rag_completeness = df["rag_completeness_normalized"].mean()
    base_completeness = df["baseline_completeness_normalized"].mean()
    rag_refusal = df["rag_refusal_correct"].mean()
    citation_coverage = df["citation_coverage"].mean()
    citation_provenance = df["citation_provenance_accuracy"].mean()
    citation_support = df["citation_support_rate"].mean()
    verified_rate = df["citation_verification_valid"].mean()
    retrieval_recall = df["retrieval_doc_recall"].mean()
    rag_latency = df["rag_latency_s"].mean()
    base_latency = df["baseline_latency_s"].mean()
    rag_wins = df["rag_wins_correctness"].mean()

    # Examples
    rag_better = df[df["rag_correctness"] > df["baseline_correctness"]]
    rag_worse = df[df["rag_correctness"] < df["baseline_correctness"]]

    lines = [
        "# A/B Evaluation Report",
        "",
        "## Setup",
        "",
        "- A: reranked RAG pipeline with enforced claim-level citation verification",
        "- B: Baseline chat model (same model, no retrieval context)",
        f"- Questions: {n}",
        "",
        "## Summary Metrics",
        "",
        f"- Correctness (RAG / baseline): {rag_correctness:.3f} / {base_correctness:.3f}",
        f"- Completeness (RAG / baseline): {rag_completeness:.3f} / {base_completeness:.3f}",
        f"- Answerable-question RAG correctness win rate: {rag_wins:.1%}",
        f"- Refusal accuracy (RAG): {rag_refusal:.1%}",
        f"- Retrieval document recall@8: {retrieval_recall:.1%}",
        f"- Claim citation coverage: {citation_coverage:.1%}",
        f"- Citation provenance accuracy: {citation_provenance:.1%}",
        f"- Semantic citation support rate: {citation_support:.1%}",
        f"- Verified-answer rate: {verified_rate:.1%}",
        f"- Avg latency RAG (s): {rag_latency:.3f}",
        f"- Avg latency Baseline (s): {base_latency:.3f}",
        "- Dataset status: curated and review-ready; not independently SME approved",
        "- Answerable questions use an LLM judge; unsupported questions use deterministic refusal scoring",
        "",
        "## Quick Conclusion",
        "",
    ]

    if rag_correctness > base_correctness:
        lines.append("RAG is better than baseline on judged factual correctness.")
    elif rag_correctness < base_correctness:
        lines.append("Baseline is better than RAG on judged factual correctness.")
    else:
        lines.append("RAG and baseline are tied on judged factual correctness.")

    lines.extend(
        [
            "",
            build_example_block(rag_better, "Examples Where RAG Is Better", limit=3),
            build_example_block(rag_worse, "Examples Where Baseline Is Better", limit=3),
        ]
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
