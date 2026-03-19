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
from src.rag import RetrievalConfig, load_resources, retrieve_candidates

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.csv"
RESULTS_PATH = EVAL_DIR / "results.csv"

BASELINE_SYSTEM_PROMPT = (
    "You are a rigorous banking regulation analyst. "
    "Answer the question as clearly as possible."
)


def baseline_answer(question: str, model: str) -> str:
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
    resources = load_resources()
    results = retrieve_candidates(
        question,
        chunks_df=resources.chunks_df,
        embedder=resources.embedder,
        index=resources.index,
        config=RetrievalConfig(top_k=top_k),
    )
    if not results:
        return ""
    ordered = list(dict.fromkeys(result.doc_id for result in results))
    return "|".join(ordered)


def main() -> None:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {QUESTIONS_PATH}")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    questions_df = pd.read_csv(QUESTIONS_PATH, encoding="utf-8")
    out_rows = []

    for row in questions_df.itertuples(index=False):
        t0 = time.time()
        rag_text = answer_question(str(row.question))
        rag_latency_s = time.time() - t0

        t1 = time.time()
        baseline_text = baseline_answer(str(row.question), model=model)
        baseline_latency_s = time.time() - t1

        out_rows.append(
            {
                "id": int(row.id),
                "question": str(row.question),
                "reference_answer": str(row.reference_answer),
                "is_answerable": int(row.is_answerable),
                "rag_answer": rag_text,
                "baseline_answer": baseline_text,
                "rag_latency_s": round(rag_latency_s, 3),
                "baseline_latency_s": round(baseline_latency_s, 3),
                "retrieved_doc_ids": doc_ids_for_question(str(row.question)),
            }
        )
        print(f"Done question {int(row.id)}")

    pd.DataFrame(out_rows).to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
