from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_pdf"
METADATA_DIR = BASE_DIR / "data" / "metadata"
METADATA_PATH = METADATA_DIR / "docs.csv"

REQUIRED_COLUMNS = [
    "doc_id",
    "doc_type",
    "topic",
    "year",
    "issue",
    "language",
    "local_path",
    "title",
]


def load_metadata(metadata_path: Path = METADATA_PATH) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    df = pd.read_csv(metadata_path, encoding="utf-8")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"docs.csv missing required columns: {missing_columns}")
    return df


def validate_metadata(df: pd.DataFrame, raw_dir: Path = RAW_DIR) -> None:
    pdf_paths = {
        str(path.resolve())
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    }
    metadata_paths = {str(Path(path).resolve()) for path in df["local_path"].tolist()}

    missing_rows = sorted(pdf_paths - metadata_paths)
    stale_rows = sorted(metadata_paths - pdf_paths)
    duplicate_doc_ids = (
        df[df["doc_id"].duplicated(keep=False)]["doc_id"].sort_values().unique().tolist()
    )

    if missing_rows:
        raise ValueError(f"PDFs missing from docs.csv: {missing_rows}")
    if stale_rows:
        raise ValueError(f"docs.csv contains missing files: {stale_rows}")
    if duplicate_doc_ids:
        raise ValueError(f"Duplicate doc_id values found: {duplicate_doc_ids}")

    if df["title"].isna().any() or (df["title"].astype(str).str.strip() == "").any():
        raise ValueError("docs.csv contains empty title values")
    if df["topic"].isna().any() or (df["topic"].astype(str).str.strip() == "").any():
        raise ValueError("docs.csv contains empty topic values")
    if df["language"].isna().any() or (df["language"].astype(str).str.strip() == "").any():
        raise ValueError("docs.csv contains empty language values")


def main() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metadata()
    validate_metadata(df)
    print(f"Validated: {METADATA_PATH}")
    print(f"Total documents: {len(df)}")


if __name__ == "__main__":
    main()
