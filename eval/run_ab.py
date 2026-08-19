"""
Simple A/B benchmark runner.

A = current RAG pipeline (src.answer.answer_question)
B = baseline chat model without retrieval context
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from src.answer import answer_question, get_client
from src.retrieve_hybrid import hybrid_search


BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "eval"
QUESTIONS_PATH = EVAL_DIR / "answer_questions.csv"
RESULTS_PATH = EVAL_DIR / "results.csv"

BASELINE_SYSTEM_PROMPT = (
    "You are a rigorous banking regulation analyst. "
    "Answer the question as clearly as possible."
)
REFUSAL_TEXTS = (
    "Les sources fournies ne permettent pas de répondre avec certitude.",
    "The provided sources do not support a sufficiently certain answer.",
)


def deterministic_unsupported_judgment(candidate: str) -> dict[str, object]:
    refused = any(marker.lower() in candidate.lower() for marker in REFUSAL_TEXTS)
    score = 4 if refused else 0
    reason = (
        "Canonical insufficient-evidence refusal."
        if refused
        else "Unsupported question received a substantive or speculative answer."
    )
    return {"correctness": score, "completeness": score, "reason": reason}


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


def judge_answer(
    question: str,
    reference: str,
    candidate: str,
    model: str,
    *,
    is_answerable: bool,
) -> dict[str, object]:
    schema = {
        "type": "object",
        "properties": {
            "correctness": {"type": "integer", "minimum": 0, "maximum": 4},
            "completeness": {"type": "integer", "minimum": 0, "maximum": 4},
            "reason": {"type": "string"},
        },
        "required": ["correctness", "completeness", "reason"],
        "additionalProperties": False,
    }
    response = get_client().chat.completions.create(
        model=model,
        temperature=0.0,
        max_completion_tokens=300,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "answer_quality", "strict": True, "schema": schema},
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "Evaluate the candidate against the reference for the stated question. "
                    "Score factual correctness and completeness from 0 to 4. Do not reward style. "
                    "When is_answerable is false, only a clear refusal or insufficient-evidence response "
                    "can receive a nonzero score; a topical answer that ignores an unsupported entity, "
                    "location, date, or qualifier must receive zero."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "reference": reference,
                        "candidate": candidate,
                        "is_answerable": is_answerable,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    return json.loads(response.choices[0].message.content or "{}")


def doc_ids_for_results(results: pd.DataFrame) -> str:
    """Get selected doc_ids from an existing retrieval result."""
    if results.empty:
        return ""
    ordered = list(dict.fromkeys(results["doc_id"].astype(str).tolist()))
    return "|".join(ordered)


def main() -> None:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {QUESTIONS_PATH}")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    judge_model = os.getenv("OPENAI_EVAL_MODEL", model)
    questions_df = pd.read_csv(QUESTIONS_PATH)
    out_rows = []

    for row in questions_df.itertuples(index=False):
        qid = int(row.id)
        question = str(row.question)
        reference_answer = str(row.reference_answer)
        is_answerable = int(row.is_answerable)

        t0 = time.time()
        retrieval_results = hybrid_search(question, top_k=8)
        rag_result = answer_question(
            question, results=retrieval_results, verify=True, return_details=True
        )
        rag_text = rag_result.answer
        rag_latency_s = time.time() - t0

        t1 = time.time()
        baseline_text = baseline_answer(question, model=model)
        baseline_latency_s = time.time() - t1

        if is_answerable:
            rag_judgment = judge_answer(
                question, reference_answer, rag_text, judge_model, is_answerable=True
            )
            baseline_judgment = judge_answer(
                question, reference_answer, baseline_text, judge_model, is_answerable=True
            )
        else:
            rag_judgment = deterministic_unsupported_judgment(rag_text)
            baseline_judgment = deterministic_unsupported_judgment(baseline_text)

        retrieved_doc_ids = doc_ids_for_results(retrieval_results)

        out_rows.append(
            {
                "id": qid,
                "question": question,
                "reference_answer": reference_answer,
                "is_answerable": is_answerable,
                "language": str(row.language),
                "category": str(row.category),
                "relevant_doc_ids": str(row.relevant_doc_ids),
                "review_status": str(row.review_status),
                "rag_answer": rag_text,
                "rag_draft_answer": rag_result.draft_answer,
                "baseline_answer": baseline_text,
                "rag_latency_s": round(rag_latency_s, 3),
                "baseline_latency_s": round(baseline_latency_s, 3),
                "retrieved_doc_ids": retrieved_doc_ids,
                "rag_correctness": int(rag_judgment["correctness"]),
                "rag_completeness": int(rag_judgment["completeness"]),
                "rag_judge_reason": str(rag_judgment["reason"]),
                "baseline_correctness": int(baseline_judgment["correctness"]),
                "baseline_completeness": int(baseline_judgment["completeness"]),
                "baseline_judge_reason": str(baseline_judgment["reason"]),
                "citation_verification_valid": int(rag_result.verification.valid),
                "citation_coverage": rag_result.verification.citation_coverage,
                "citation_provenance_accuracy": rag_result.verification.provenance_accuracy,
                "citation_support_rate": rag_result.verification.support_rate,
                "citation_verification_errors": json.dumps(
                    rag_result.verification.errors, ensure_ascii=False
                ),
            }
        )
        print(f"Done question {qid}")

    results_df = pd.DataFrame(out_rows)
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
