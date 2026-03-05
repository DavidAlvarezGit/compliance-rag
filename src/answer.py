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


def build_context(results):
    context_blocks = []
    for _, row in results.iterrows():
        block = f"""
Source: {row['doc_id']} (pp. {row['page_start']}-{row['page_end']})
{row['chunk_text']}
"""
        context_blocks.append(block.strip())
    return "\n\n".join(context_blocks)


def answer_question(query):
    try:
        from .retrieve_hybrid import hybrid_search
    except ImportError:
        from retrieve_hybrid import hybrid_search

    results = hybrid_search(query, top_k=8)
    if results.empty:
        return "Les sources fournies ne permettent pas de repondre avec certitude."

    context = build_context(results)

    prompt = f"""
You are a professional banking regulation analyst specialized in the Basel III framework.

Your task:
- Answer strictly using ONLY the provided sources.
- If the sources are insufficient, say exactly:
  "Les sources fournies ne permettent pas de repondre avec certitude."
- Do NOT use outside knowledge.
- Do NOT generalize beyond what is written.
- Do not infer or extrapolate beyond what is explicitly written.

Output format (strictly follow this structure):

REPONSE SYNTHETIQUE:
(3-6 bullet points maximum, concise and factual)

ANALYSE DETAILLEE:
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
            {"role": "system", "content": "You are a rigorous banking regulation analyst. You must never hallucinate."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_completion_tokens=500,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    question = "What are the operational risk resilience requirements?"
    answer = answer_question(question)
    print(answer)
