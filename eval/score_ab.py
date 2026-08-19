"""
Simple scorer for eval/results.csv.

The goal is beginner-friendly metrics:
- keyword recall vs reference answer (proxy for correctness)
- refusal behavior on unanswerable questions
- citation presence (for RAG groundedness)
- latency comparison
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "eval" / "results.csv"
SCORED_PATH = BASE_DIR / "eval" / "results_scored.csv"

REFUSAL_TEXTS = (
    "Les sources fournies ne permettent pas de répondre avec certitude.",
    "The provided sources do not support a sufficiently certain answer.",
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "into",
    "about",
    "dans",
    "avec",
    "pour",
    "des",
    "une",
    "les",
    "sur",
    "est",
    "sont",
}


def normalize_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9àâçéèêëîïôûùüÿñæœ]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def keyword_recall(reference: str, candidate: str) -> float:
    ref_tokens = normalize_tokens(reference)
    if not ref_tokens:
        return 0.0
    cand_tokens = normalize_tokens(candidate)
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


def has_citation(text: str) -> int:
    if re.search(r"\(Source:\s*.+?pp\.\s*\d+\-\d+\)", text or "", flags=re.IGNORECASE):
        return 1
    return 0


def refused(text: str) -> int:
    normalized = (text or "").lower()
    return int(any(refusal.lower() in normalized for refusal in REFUSAL_TEXTS))


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {RESULTS_PATH}")

    df = pd.read_csv(RESULTS_PATH)

    df["rag_keyword_recall"] = df.apply(
        lambda r: keyword_recall(str(r["reference_answer"]), str(r["rag_answer"]))
        if int(r["is_answerable"]) == 1
        else float("nan"),
        axis=1,
    )
    df["baseline_keyword_recall"] = df.apply(
        lambda r: keyword_recall(str(r["reference_answer"]), str(r["baseline_answer"]))
        if int(r["is_answerable"]) == 1
        else float("nan"),
        axis=1,
    )

    df["rag_has_citation"] = df.apply(
        lambda r: has_citation(str(r["rag_answer"]))
        if int(r["is_answerable"]) == 1
        else float("nan"),
        axis=1,
    )
    df["rag_refused"] = df["rag_answer"].apply(lambda x: refused(str(x)))

    df["rag_refusal_correct"] = df.apply(
        lambda r: int(r["rag_refused"] == (1 if int(r["is_answerable"]) == 0 else 0)),
        axis=1,
    )
    df["rag_win_keyword"] = df.apply(
        lambda r: int(r["rag_keyword_recall"] > r["baseline_keyword_recall"])
        if int(r["is_answerable"]) == 1
        else float("nan"),
        axis=1,
    )
    df["tie_keyword"] = df.apply(
        lambda r: int(r["rag_keyword_recall"] == r["baseline_keyword_recall"])
        if int(r["is_answerable"]) == 1
        else float("nan"),
        axis=1,
    )

    df.to_csv(SCORED_PATH, index=False, encoding="utf-8")

    print("=== A/B Score Summary ===")
    print(f"Questions: {len(df)}")
    print(f"Avg keyword recall (RAG):      {df['rag_keyword_recall'].mean():.3f}")
    print(f"Avg keyword recall (Baseline): {df['baseline_keyword_recall'].mean():.3f}")
    print(f"RAG wins by keyword recall:    {df['rag_win_keyword'].mean():.1%}")
    print(f"Ties by keyword recall:        {df['tie_keyword'].mean():.1%}")
    print(f"Refusal accuracy (RAG):        {df['rag_refusal_correct'].mean():.1%}")
    print(f"RAG citation presence:         {df['rag_has_citation'].mean():.1%}")
    print(f"Avg latency (RAG, s):          {df['rag_latency_s'].mean():.3f}")
    print(f"Avg latency (Baseline, s):     {df['baseline_latency_s'].mean():.3f}")
    print(f"\nSaved: {SCORED_PATH}")


if __name__ == "__main__":
    main()
