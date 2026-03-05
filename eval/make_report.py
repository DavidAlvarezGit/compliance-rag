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
            f"**Keyword recall:** RAG={float(row.rag_keyword_recall):.3f}, "
            f"Baseline={float(row.baseline_keyword_recall):.3f}"
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

    rag_recall = df["rag_keyword_recall"].mean()
    base_recall = df["baseline_keyword_recall"].mean()
    rag_refusal = df["rag_refusal_correct"].mean()
    base_refusal = df["baseline_refusal_correct"].mean()
    rag_citation = df["rag_has_citation"].mean()
    rag_latency = df["rag_latency_s"].mean()
    base_latency = df["baseline_latency_s"].mean()
    rag_wins = (df["rag_keyword_recall"] > df["baseline_keyword_recall"]).mean()
    ties = (df["rag_keyword_recall"] == df["baseline_keyword_recall"]).mean()

    # Examples
    rag_better = df[df["rag_keyword_recall"] > df["baseline_keyword_recall"]]
    rag_worse = df[df["rag_keyword_recall"] < df["baseline_keyword_recall"]]

    lines = [
        "# A/B Evaluation Report",
        "",
        "## Setup",
        "",
        "- A: RAG pipeline (`src.answer.answer_question`)",
        "- B: Baseline chat model (same model, no retrieval context)",
        f"- Questions: {n}",
        "",
        "## Summary Metrics",
        "",
        f"- Avg keyword recall (RAG): {rag_recall:.3f}",
        f"- Avg keyword recall (Baseline): {base_recall:.3f}",
        f"- RAG win rate (keyword recall): {rag_wins:.1%}",
        f"- Tie rate: {ties:.1%}",
        f"- Refusal accuracy (RAG): {rag_refusal:.1%}",
        f"- Refusal accuracy (Baseline): {base_refusal:.1%}",
        f"- RAG citation presence: {rag_citation:.1%}",
        f"- Avg latency RAG (s): {rag_latency:.3f}",
        f"- Avg latency Baseline (s): {base_latency:.3f}",
        "",
        "## Quick Conclusion",
        "",
    ]

    if rag_recall > base_recall:
        lines.append("RAG is better than baseline on this dataset by keyword-recall proxy.")
    elif rag_recall < base_recall:
        lines.append("Baseline is better than RAG on this dataset by keyword-recall proxy.")
    else:
        lines.append("RAG and baseline are tied on this dataset by keyword-recall proxy.")

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
