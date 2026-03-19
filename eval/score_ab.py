"""
Simple scorer for eval/results.csv.

The goal is beginner-friendly metrics:
- keyword recall vs reference answer (proxy for correctness)
- refusal behavior on unanswerable questions
- citation presence (for RAG groundedness)
- latency comparison
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval_metrics import has_citation, is_refusal, keyword_recall

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "eval" / "results.csv"
SCORED_PATH = BASE_DIR / "eval" / "results_scored.csv"


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RESULTS_PATH}")

    df = pd.read_csv(RESULTS_PATH, encoding="utf-8")
    df["rag_keyword_recall"] = df.apply(
        lambda row: keyword_recall(str(row["reference_answer"]), str(row["rag_answer"])),
        axis=1,
    )
    df["baseline_keyword_recall"] = df.apply(
        lambda row: keyword_recall(str(row["reference_answer"]), str(row["baseline_answer"])),
        axis=1,
    )
    df["rag_has_citation"] = df["rag_answer"].apply(lambda text: has_citation(str(text)))
    df["rag_refused"] = df["rag_answer"].apply(lambda text: is_refusal(str(text)))
    df["baseline_refused"] = df["baseline_answer"].apply(lambda text: is_refusal(str(text)))
    df["rag_refusal_correct"] = df.apply(
        lambda row: int(row["rag_refused"] == (1 if int(row["is_answerable"]) == 0 else 0)),
        axis=1,
    )
    df["baseline_refusal_correct"] = df.apply(
        lambda row: int(row["baseline_refused"] == (1 if int(row["is_answerable"]) == 0 else 0)),
        axis=1,
    )
    df["rag_win_keyword"] = (df["rag_keyword_recall"] > df["baseline_keyword_recall"]).astype(int)
    df["tie_keyword"] = (df["rag_keyword_recall"] == df["baseline_keyword_recall"]).astype(int)

    df.to_csv(SCORED_PATH, index=False, encoding="utf-8")

    print("=== A/B Score Summary ===")
    print(f"Questions: {len(df)}")
    print(f"Avg keyword recall (RAG):      {df['rag_keyword_recall'].mean():.3f}")
    print(f"Avg keyword recall (Baseline): {df['baseline_keyword_recall'].mean():.3f}")
    print(f"RAG wins by keyword recall:    {df['rag_win_keyword'].mean():.1%}")
    print(f"Ties by keyword recall:        {df['tie_keyword'].mean():.1%}")
    print(f"Refusal accuracy (RAG):        {df['rag_refusal_correct'].mean():.1%}")
    print(f"Refusal accuracy (Baseline):   {df['baseline_refusal_correct'].mean():.1%}")
    print(f"RAG citation presence:         {df['rag_has_citation'].mean():.1%}")
    print(f"Avg latency (RAG, s):          {df['rag_latency_s'].mean():.3f}")
    print(f"Avg latency (Baseline, s):     {df['baseline_latency_s'].mean():.3f}")
    print(f"\nSaved: {SCORED_PATH}")


if __name__ == "__main__":
    main()
