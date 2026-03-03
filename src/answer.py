from retrieve_hybrid import hybrid_search
from dotenv import load_dotenv
from openai import OpenAI
import os

# --- Load environment variables ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
def answer_question(query):

    # 🔒 Reduce cost: fewer chunks
    results = hybrid_search(query, top_k=8)

    context = build_context(results)

    prompt = f"""
You are an economic research assistant specialized in SNB publications.

Answer the question strictly using ONLY the provided sources.
If the sources do not contain enough information, explicitly say:
"Les sources fournies ne permettent pas de répondre avec certitude."

For each factual statement, cite the source exactly in this format:
(Source: DOC_ID pp.X-Y)

Do not invent information.
Do not use outside knowledge.

Question:
{query}

Sources:
{context}
"""

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.1,
    max_completion_tokens=600
)

    return response.choices[0].message.content


if __name__ == "__main__":
    question = "Quels sont les principaux risques pesant sur la croissance suisse récemment ?"
    answer = answer_question(question)
    print(answer)