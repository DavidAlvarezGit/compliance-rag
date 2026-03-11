from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PAGES_PATH = BASE_DIR / "data" / "processed" / "pages.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"
TEMP_OUTPUT_PATH = OUTPUT_PATH.with_suffix(".tmp.parquet")

TARGET_CHARS = 1600
MAX_CHARS = 2200
MIN_CHARS = 600
OVERLAP_PARAGRAPHS = 1


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"Bulletin trimestriel.*?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    parts = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = [line.strip() for line in part.splitlines() if line.strip()]
        merged = " ".join(lines)
        merged = re.sub(r"\s+", " ", merged).strip()
        if merged:
            paragraphs.append(merged)
    return paragraphs


def split_long_paragraph(paragraph: str, max_chars: int = MAX_CHARS) -> list[str]:
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars:
        return [paragraph] if paragraph else []

    sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        if len(sentence) <= max_chars:
            buffer = sentence
            continue
        for start in range(0, len(sentence), max_chars):
            chunks.append(sentence[start : start + max_chars].strip())
        buffer = ""
    if buffer:
        chunks.append(buffer)
    return [chunk for chunk in chunks if chunk]


def paragraph_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in split_paragraphs(text):
        units.extend(split_long_paragraph(paragraph))
    return units


def make_chunk_record(group: pd.DataFrame, page_start: int, page_end: int, text: str) -> dict[str, object]:
    first_row = group.iloc[0]
    return {
        "doc_id": first_row["doc_id"],
        "doc_type": first_row["doc_type"],
        "topic": first_row.get("topic", None),
        "year": first_row["year"],
        "issue": first_row.get("issue", None),
        "language": first_row["language"],
        "page_start": int(page_start),
        "page_end": int(page_end),
        "chunk_text": text.strip(),
    }


def build_chunks_for_doc(group: pd.DataFrame) -> list[dict[str, object]]:
    units: list[tuple[int, str]] = []
    for _, row in group.iterrows():
        for unit in paragraph_units(row["text"]):
            units.append((int(row["page"]), unit))

    if not units:
        return []

    chunks: list[dict[str, object]] = []
    start = 0
    while start < len(units):
        end = start
        chunk_parts: list[str] = []
        page_start = units[start][0]
        page_end = page_start

        while end < len(units):
            page, unit = units[end]
            candidate = "\n\n".join(chunk_parts + [unit]) if chunk_parts else unit
            if chunk_parts and len(candidate) > MAX_CHARS:
                break
            chunk_parts.append(unit)
            page_end = page
            end += 1
            if len(candidate) >= TARGET_CHARS:
                break

        if not chunk_parts:
            start += 1
            continue

        chunk_text = "\n\n".join(chunk_parts).strip()
        if len(chunk_text) >= MIN_CHARS:
            chunks.append(
                make_chunk_record(
                    group=group,
                    page_start=page_start,
                    page_end=page_end,
                    text=chunk_text,
                )
            )

        if end >= len(units):
            break

        overlap = min(OVERLAP_PARAGRAPHS, len(chunk_parts) - 1)
        start = end - overlap if overlap > 0 else end

    return [chunk for chunk in chunks if len(str(chunk["chunk_text"]).strip()) >= MIN_CHARS]


def main() -> None:
    pages_df = pd.read_parquet(PAGES_PATH).copy()
    pages_df["text"] = pages_df["text"].map(clean_text)
    pages_df = pages_df[pages_df["text"].astype(bool)].copy()

    chunks: list[dict[str, object]] = []
    for _, group in pages_df.groupby("doc_id", sort=True):
        group = group.sort_values("page").reset_index(drop=True)
        chunks.extend(build_chunks_for_doc(group))

    chunks_df = pd.DataFrame(chunks)
    chunks_df.to_parquet(TEMP_OUTPUT_PATH, index=False)
    TEMP_OUTPUT_PATH.replace(OUTPUT_PATH)

    print(f"Chunks created: {len(chunks_df)}")
    if not chunks_df.empty:
        print(f"Documents chunked: {chunks_df['doc_id'].nunique()}")
        print(f"Average chunk chars: {chunks_df['chunk_text'].str.len().mean():.1f}")


if __name__ == "__main__":
    main()
