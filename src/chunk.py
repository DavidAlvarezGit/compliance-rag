from pathlib import Path
import pandas as pd
import re

BASE_DIR = Path(__file__).resolve().parent.parent
PAGES_PATH = BASE_DIR / "data" / "processed" / "pages.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"

df = pd.read_parquet(PAGES_PATH)

def clean_text(text):
    # remove repeated bulletin header
    text = re.sub(r"Bulletin trimestriel.*?\n", "", text)
    # remove page number at beginning
    text = re.sub(r"^\d+\n", "", text)
    # normalize whitespace
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

chunks = []

MAX_CHARS = 4000   # simple char-based chunking (safe for MVP)

for doc_id, group in df.groupby("doc_id"):

    group = group.sort_values("page")

    buffer = ""
    page_start = None

    for _, row in group.iterrows():

        text = row["clean_text"]

        if not text:
            continue

        if page_start is None:
            page_start = row["page"]

        if len(buffer) + len(text) < MAX_CHARS:
            buffer += "\n" + text
        else:
            chunks.append({
                "doc_id": row["doc_id"],
                "doc_type": row["doc_type"],
                "year": row["year"],
                "issue": row["issue"],
                "language": row["language"],
                "page_start": page_start,
                "page_end": row["page"],
                "chunk_text": buffer.strip()
            })
            buffer = text
            page_start = row["page"]

    if buffer:
        chunks.append({
            "doc_id": row["doc_id"],
            "doc_type": row["doc_type"],
            "year": row["year"],
            "issue": row["issue"],
            "language": row["language"],
            "page_start": page_start,
            "page_end": row["page"],
            "chunk_text": buffer.strip()
        })

df_chunks = pd.DataFrame(chunks)
df_chunks.to_parquet(OUTPUT_PATH, index=False)

print("Chunks created:", len(df_chunks))