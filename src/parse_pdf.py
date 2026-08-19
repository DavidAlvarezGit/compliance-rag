from __future__ import annotations

from pathlib import Path

import pandas as pd
import fitz  # pymupdf

try:
    from .metadata import resolve_local_path
except ImportError:
    from metadata import resolve_local_path

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = BASE_DIR / "data" / "metadata" / "docs.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "pages.parquet"
TEMP_OUTPUT_PATH = OUTPUT_DIR / "pages.tmp.parquet"


def parse_documents(df_docs: pd.DataFrame) -> pd.DataFrame:
    pages: list[dict[str, object]] = []
    for _, row in df_docs.iterrows():
        pdf_path = resolve_local_path(row["local_path"])
        print(f"Parsing {pdf_path.name}...")

        with fitz.open(pdf_path) as doc:
            for page_number, page in enumerate(doc):
                text = page.get_text("text").strip()
                if not text:
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
    df_docs = pd.read_csv(METADATA_PATH)
    df_pages = parse_documents(df_docs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_pages.to_parquet(TEMP_OUTPUT_PATH, index=False)
    TEMP_OUTPUT_PATH.replace(OUTPUT_PATH)
    print("Saved pages.parquet")
    print("Total pages:", len(df_pages))


if __name__ == "__main__":
    main()
