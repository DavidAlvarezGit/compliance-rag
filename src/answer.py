from pathlib import Path
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .citations import VerificationReport, verify_citations
except ImportError:
    from citations import VerificationReport, verify_citations

# --- Load environment variables ---
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_VERIFIER_MODEL = os.getenv("OPENAI_VERIFIER_MODEL", OPENAI_MODEL)


@dataclass
class AnswerResult:
    answer: str
    draft_answer: str
    verification: VerificationReport


def select_verified_answer(
    draft: str, report: VerificationReport, query: str
) -> str:
    """Return the full draft, a claim-filtered answer, or a safe refusal."""
    if report.valid:
        return draft
    if report.can_return_partial:
        return "\n\n".join(claim.text for claim in report.verified_claims)
    return insufficient_evidence_message(query)


def insufficient_evidence_message(query: str) -> str:
    french_markers = {"le", "la", "les", "des", "une", "quelles", "quel", "dans", "sur"}
    words = {word.strip(".,?!:;").lower() for word in query.split()}
    if words & french_markers or any(char in query.lower() for char in "àâçéèêëîïôùûüÿœ"):
        return "Les sources fournies ne permettent pas de répondre avec certitude."
    return "The provided sources do not support a sufficiently certain answer."


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def build_context(results):
    context_blocks = []
    for _, row in results.iterrows():
        block = f"""
Source: {row['doc_id']} (pp. {row['page_start']}-{row['page_end']})
{row['chunk_text']}
"""
        context_blocks.append(block.strip())
    return "\n\n".join(context_blocks)


def answer_question(query, results=None, *, verify=True, return_details=False):
    try:
        from .retrieve_hybrid import hybrid_search
    except ImportError:
        from retrieve_hybrid import hybrid_search

    if results is None:
        results = hybrid_search(query, top_k=8)
    if results.empty:
        answer = insufficient_evidence_message(query)
        report = verify_citations(answer, results)
        result = AnswerResult(answer=answer, draft_answer=answer, verification=report)
        return result if return_details else result.answer

    context = build_context(results)

    prompt = f"""
You are a professional banking regulation analyst focused on Swiss banking regulation and the Basel III framework.

Your task:
- Answer strictly using ONLY the provided sources.
- Answer in the same language as the user's question.
- If the sources are insufficient, clearly refuse in the same language as the question.
- Unless the user explicitly names another jurisdiction, interpret the question from a Swiss regulatory perspective.
- Treat a cited Swiss law, FINMA instrument, or FMIA provision as sufficient Swiss jurisdictional context; do not require every claim to repeat "Switzerland".
- Basel standards are international standards. Do not present them as binding Swiss law unless the supplied sources support their Swiss implementation.
- Preserve the scope of the question and do not invent missing dates, jurisdictions, entities, products, or conditions.
- Context established by the question or cited legal instrument does not need to be repeated in every claim.
- Do NOT use outside knowledge.
- Use faithful paraphrases and direct conclusions, but do not add unsupported details.

Output requirements:
- Answer with the minimum number of claims needed. Prefer 1-3 short paragraphs; use a fourth only when the question has distinct parts.
- Each paragraph MUST contain exactly one factual sentence followed immediately by its citation.
- Use bullets only for a genuine list of separate requirements or conditions. Each bullet must follow the same one-sentence citation rule.
- Stop when the question is directly answered. Do not repeat or expand the answer merely because more evidence is available.
- Do not write introductory text, conclusions, headings, or uncited factual statements.
- Do not combine separately sourced claims in one sentence.
- Use this exact citation format: (Source: DOC_ID pp.X-Y)
- Copy DOC_ID and the page range exactly from the supplied source header.

If multiple sources support a claim, cite multiple sources.


Question:
{query}

Sources:
{context}
"""

    client = get_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous banking regulation analyst. "
                    "You must never hallucinate. "
                    "Always answer in the same language as the user's question."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_completion_tokens=1100,
    )

    draft = response.choices[0].message.content or ""
    report = verify_citations(
        draft,
        results,
        client=client,
        model=OPENAI_VERIFIER_MODEL,
        semantic=verify,
        question=query,
    )
    answer = select_verified_answer(draft, report, query)
    result = AnswerResult(answer=answer, draft_answer=draft, verification=report)
    return result if return_details else result.answer


if __name__ == "__main__":
    question = "What are the operational risk resilience requirements?"
    answer = answer_question(question)
    print(answer)
