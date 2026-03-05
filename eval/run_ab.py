"""
Simple A/B benchmark runner.

A = current RAG pipeline (src.answer.answer_question)
B = baseline chat model without retrieval context
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from src.answer import answer_question, get_client
from src.retrieve_hybrid import hybrid_search


BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.csv"
RESULTS_PATH = EVAL_DIR / "results.csv"

BASELINE_SYSTEM_PROMPT = (
    "You are a rigorous banking regulation analyst. "
    "Answer the question as clearly as possible."
)


def baseline_answer(question: str, model: str) -> str:
    """Call the same LLM model but without retrieval context."""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_completion_tokens=500,
    )
    return response.choices[0].message.content or ""


def doc_ids_for_question(question: str, top_k: int = 8) -> str:
    """Get selected doc_ids from hybrid retrieval for traceability."""
    results = hybrid_search(question, top_k=top_k)
    if results.empty:
        return ""
    ordered = list(dict.fromkeys(results["doc_id"].astype(str).tolist()))
    return "|".join(ordered)


def main() -> None:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {QUESTIONS_PATH}")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    questions_df = pd.read_csv(QUESTIONS_PATH)
    out_rows = []

    for row in questions_df.itertuples(index=False):
        qid = int(row.id)
        question = str(row.question)
        reference_answer = str(row.reference_answer)
        is_answerable = int(row.is_answerable)

        t0 = time.time()
        rag_text = answer_question(question)
        rag_latency_s = time.time() - t0

        t1 = time.time()
        baseline_text = baseline_answer(question, model=model)
        baseline_latency_s = time.time() - t1

        retrieved_doc_ids = doc_ids_for_question(question)

        out_rows.append(
            {
                "id": qid,
                "question": question,
                "reference_answer": reference_answer,
                "is_answerable": is_answerable,
                "rag_answer": rag_text,
                "baseline_answer": baseline_text,
                "rag_latency_s": round(rag_latency_s, 3),
                "baseline_latency_s": round(baseline_latency_s, 3),
                "retrieved_doc_ids": retrieved_doc_ids,
            }
        )
        print(f"Done question {qid}")

    results_df = pd.DataFrame(out_rows)
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
