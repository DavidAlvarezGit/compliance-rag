"""
Score answer quality, refusal behavior, grounding, retrieval, and latency.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from eval.retrieval_metrics import parse_doc_ids, recall_at_k


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "eval" / "results.csv"
SCORED_PATH = BASE_DIR / "eval" / "results_scored.csv"
SUMMARY_PATH = BASE_DIR / "eval" / "answer_summary.json"
THRESHOLDS_PATH = BASE_DIR / "eval" / "answer_thresholds.json"

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


def _check(summary: dict[str, float]) -> None:
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    failures = []
    for metric, minimum in thresholds["minimums"].items():
        if summary[metric] < float(minimum):
            failures.append(f"{metric}={summary[metric]:.3f} < {minimum:.3f}")
    allowed_drop = float(thresholds["maximum_correctness_drop_from_baseline"])
    delta = summary["rag_correctness"] - summary["baseline_correctness"]
    if delta < -allowed_drop:
        failures.append(f"correctness delta={delta:+.3f} exceeds allowed drop {allowed_drop:.3f}")
    if failures:
        raise SystemExit("Answer-quality regression gate failed: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail when configured thresholds regress.")
    args = parser.parse_args()
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

    answerable = df["is_answerable"].astype(int) == 1
    df["rag_has_citation"] = (df["citation_coverage"] > 0).where(answerable)
    df["rag_refused"] = df["rag_answer"].apply(lambda x: refused(str(x)))
    df["baseline_refused"] = df["baseline_answer"].apply(lambda x: refused(str(x)))

    # Unsupported cases use a deterministic safety rubric; LLM judges were not
    # consistent when scoring identical refusals.
    unsupported = ~answerable
    df.loc[unsupported, "rag_correctness"] = df.loc[unsupported, "rag_refused"] * 4
    df.loc[unsupported, "rag_completeness"] = df.loc[unsupported, "rag_refused"] * 4
    df.loc[unsupported, "baseline_correctness"] = df.loc[unsupported, "baseline_refused"] * 4
    df.loc[unsupported, "baseline_completeness"] = df.loc[unsupported, "baseline_refused"] * 4

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
    df["retrieval_doc_recall"] = df.apply(
        lambda r: recall_at_k(
            parse_doc_ids(r["retrieved_doc_ids"]), parse_doc_ids(r["relevant_doc_ids"]), 8
        )
        if int(r["is_answerable"]) == 1
        else float("nan"),
        axis=1,
    )
    df["rag_correctness_normalized"] = df["rag_correctness"] / 4.0
    df["baseline_correctness_normalized"] = df["baseline_correctness"] / 4.0
    df["rag_completeness_normalized"] = df["rag_completeness"] / 4.0
    df["baseline_completeness_normalized"] = df["baseline_completeness"] / 4.0
    df["rag_wins_correctness"] = (df["rag_correctness"] > df["baseline_correctness"]).astype(int)
    df["rag_wins_correctness"] = df["rag_wins_correctness"].where(answerable)
    for column in (
        "citation_coverage",
        "citation_provenance_accuracy",
        "citation_support_rate",
        "citation_verification_valid",
    ):
        df[column] = df[column].where(answerable)

    df.to_csv(SCORED_PATH, index=False, encoding="utf-8")

    summary = {
        "questions": int(len(df)),
        "answerable_questions": int(answerable.sum()),
        "unsupported_questions": int((~answerable).sum()),
        "rag_correctness": float(df["rag_correctness_normalized"].mean()),
        "baseline_correctness": float(df["baseline_correctness_normalized"].mean()),
        "rag_completeness": float(df["rag_completeness_normalized"].mean()),
        "baseline_completeness": float(df["baseline_completeness_normalized"].mean()),
        "answerable_rag_correctness": float(
            df.loc[answerable, "rag_correctness_normalized"].mean()
        ),
        "answerable_baseline_correctness": float(
            df.loc[answerable, "baseline_correctness_normalized"].mean()
        ),
        "unsupported_refusal_accuracy": float(df.loc[unsupported, "rag_refused"].mean()),
        "unsupported_baseline_refusal_accuracy": float(
            df.loc[unsupported, "baseline_refused"].mean()
        ),
        "refusal_accuracy": float(df["rag_refusal_correct"].mean()),
        "retrieval_recall_at_8": float(df["retrieval_doc_recall"].mean()),
        "citation_coverage": float(df["citation_coverage"].mean()),
        "citation_provenance_accuracy": float(df["citation_provenance_accuracy"].mean()),
        "citation_support_rate": float(df["citation_support_rate"].mean()),
        "verified_answer_rate": float(df["citation_verification_valid"].mean()),
        "answerable_rag_win_rate": float(df["rag_wins_correctness"].mean()),
        "rag_latency_mean_s": float(df["rag_latency_s"].mean()),
        "baseline_latency_mean_s": float(df["baseline_latency_s"].mean()),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("=== A/B Score Summary ===")
    print(f"Questions: {len(df)}")
    print(f"Correctness (RAG):             {df['rag_correctness_normalized'].mean():.3f}")
    print(f"Correctness (Baseline):        {df['baseline_correctness_normalized'].mean():.3f}")
    print(f"Completeness (RAG):            {df['rag_completeness_normalized'].mean():.3f}")
    print(f"Completeness (Baseline):       {df['baseline_completeness_normalized'].mean():.3f}")
    print(f"Answerable RAG win rate:       {df['rag_wins_correctness'].mean():.1%}")
    print(f"Unsupported refusal accuracy:  {df.loc[unsupported, 'rag_refused'].mean():.1%}")
    print(f"Refusal accuracy (RAG):        {df['rag_refusal_correct'].mean():.1%}")
    print(f"Retrieval document recall@8:   {df['retrieval_doc_recall'].mean():.1%}")
    print(f"Citation coverage:             {df['citation_coverage'].mean():.1%}")
    print(f"Citation provenance accuracy:  {df['citation_provenance_accuracy'].mean():.1%}")
    print(f"Citation support rate:         {df['citation_support_rate'].mean():.1%}")
    print(f"Verified-answer rate:          {df['citation_verification_valid'].mean():.1%}")
    print(f"Avg latency (RAG, s):          {df['rag_latency_s'].mean():.3f}")
    print(f"Avg latency (Baseline, s):     {df['baseline_latency_s'].mean():.3f}")
    print(f"\nSaved: {SCORED_PATH}")
    if args.check:
        _check(summary)


if __name__ == "__main__":
    main()
