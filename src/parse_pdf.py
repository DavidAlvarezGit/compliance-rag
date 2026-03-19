from __future__ import annotations

from pathlib import Path

import fitz
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_DIR / "data" / "metadata" / "docs.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "pages.parquet"


def parse_all_pdfs(metadata_path: Path = METADATA_PATH) -> pd.DataFrame:
    df_docs = pd.read_csv(metadata_path, encoding="utf-8")
    pages: list[dict[str, object]] = []

    for _, row in df_docs.iterrows():
        pdf_path = Path(row["local_path"])
        print(f"Parsing {pdf_path.name}...")
        doc = fitz.open(pdf_path)
        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            text = page.get_text("text").strip()
            if len(text) < 50:
                continue
            pages.append(
                {
                    "doc_id": row["doc_id"],
                    "doc_type": row["doc_type"],
                    "topic": row.get("topic", None),
                    "year": row["year"],
                    "issue": row.get("issue", None),
                    "language": row["language"],
                    "page": page_number + 1,
                    "text": text,
                }
            )
    return pd.DataFrame(pages)


def main() -> None:
    df_pages = parse_all_pdfs()
    df_pages.to_parquet(OUTPUT_PATH, index=False)
    print("Saved pages.parquet")
    print("Total pages:", len(df_pages))


if __name__ == "__main__":
    main()
