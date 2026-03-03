from pathlib import Path
import pandas as pd
import fitz  # pymupdf

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_DIR / "data" / "metadata" / "docs.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_docs = pd.read_csv(METADATA_PATH)

pages = []

for _, row in df_docs.iterrows():

    pdf_path = Path(row["local_path"])

    print(f"Parsing {pdf_path.name}...")

    doc = fitz.open(pdf_path)

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text("text")

        # Basic cleaning
        text = text.strip()
        if len(text) < 50:
            continue

        pages.append({
            "doc_id": row["doc_id"],
            "doc_type": row["doc_type"],
            "year": row["year"],
            "issue": row["issue"],
            "language": row["language"],
            "page": page_number + 1,
            "text": text
        })

df_pages = pd.DataFrame(pages)

df_pages.to_parquet(OUTPUT_DIR / "pages.parquet", index=False)

print("Saved pages.parquet")
print("Total pages:", len(df_pages))