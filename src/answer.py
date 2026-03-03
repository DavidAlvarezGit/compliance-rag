from pathlib import Path
from retrieve_hybrid import hybrid_search
import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def build_context(results):
    context_blocks = []
    for i, row in results.iterrows():
        block = f"""
Source: {row['doc_id']} (pp. {row['page_start']}-{row['page_end']})
{row['chunk_text']}
"""
        context_blocks.append(block)
    return "\n\n".join(context_blocks)

def answer_question(query):

    results = hybrid_search(query, top_k=8)
    context = build_context(results)

    prompt = f"""
You are an economic research assistant specialized in SNB publications.

Answer the question strictly using the provided sources.
If the sources do not contain enough information, say so.

For each factual statement, cite the source in this format:
(Source: DOC_ID pp.X-Y)

Question:
{query}

Sources:
{context}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # or your preferred model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    question = "Quels sont les principaux risques pesant sur la croissance suisse récemment ?"
    answer = answer_question(question)
    print(answer)