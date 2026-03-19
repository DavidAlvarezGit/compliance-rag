from __future__ import annotations

try:
    from .rag import answer_question, build_context, generate_answer, load_openai_client
except ImportError:
    from rag import answer_question, build_context, generate_answer, load_openai_client


def get_client():
    return load_openai_client()


if __name__ == "__main__":
    question = "What are the operational risk resilience requirements?"
    answer = answer_question(question)
    print(answer)
