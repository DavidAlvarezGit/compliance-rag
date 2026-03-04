from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI

# --- Load environment variables ---
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)

# --- Build context from retrieved chunks ---
def build_context(results):
    context_blocks = []
    for _, row in results.iterrows():
        block = f"""
Source: {row['doc_id']} (pp. {row['page_start']}-{row['page_end']})
{row['chunk_text']}
"""
        context_blocks.append(block.strip())
    return "\n\n".join(context_blocks)

# --- Main answering function ---
def answer_question(query, min_year=None):
    try:
        from .retrieve_hybrid import hybrid_search
    except ImportError:
        from retrieve_hybrid import hybrid_search

    # --- Retrieval ---
    results = hybrid_search(query, top_k=8)

    # Optional recency filter
    if min_year is not None:
        results = results[results["year"] >= min_year]

    if results.empty:
        return "Les sources fournies ne permettent pas de répondre avec certitude."

    context = build_context(results)

    prompt = f"""
You are a professional macroeconomic research analyst specialized in SNB publications.

Your task:
- Answer strictly using ONLY the provided sources.
- If the sources are insufficient, say exactly:
  "Les sources fournies ne permettent pas de répondre avec certitude."
- Do NOT use outside knowledge.
- Do NOT generalize beyond what is written.
- If the question includes a time reference (e.g., "récemment"), prioritize the most recent sources in your answer.
- If you use older sources, explicitly label them as "contexte historique" and keep them secondary.
- Do not infer or extrapolate beyond what is explicitly written.

Output format (strictly follow this structure):

RÉPONSE SYNTHÉTIQUE:
(3–6 bullet points maximum, concise and factual)

ANALYSE DÉTAILLÉE:
(Short structured paragraphs explaining each risk)

Each factual statement MUST include a citation in this format:
(Source: DOC_ID pp.X-Y)

If multiple sources support a claim, cite multiple sources.


Question:
{query}

Sources:
{context}
"""

    response = get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a rigorous economic analyst. You must never hallucinate."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # more deterministic
        max_completion_tokens=500
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    question = "Quels sont les principaux risques pesant sur la croissance suisse récemment ?"
    answer = answer_question(question)
    print(answer)
