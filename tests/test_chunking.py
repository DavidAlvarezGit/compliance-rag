import pandas as pd

from src.chunk import build_chunks_for_doc


def test_short_final_section_is_preserved() -> None:
    final_text = "FINAL-OBLIGATION " + "z" * 90
    text = "\n\n".join(("a" * 900, "b" * 800, final_text))
    pages = pd.DataFrame(
        [
            {
                "doc_id": "doc",
                "doc_type": "REG_BANK",
                "topic": "test",
                "year": 2026,
                "issue": None,
                "language": "EN",
                "page": 1,
                "text": text,
            }
        ]
    )

    chunks = build_chunks_for_doc(pages)

    assert chunks
    assert any(final_text in str(chunk["chunk_text"]) for chunk in chunks)


def test_tiny_document_remains_searchable() -> None:
    pages = pd.DataFrame(
        [
            {
                "doc_id": "short",
                "doc_type": "REG_BANK",
                "topic": "test",
                "year": 2026,
                "issue": None,
                "language": "EN",
                "page": 1,
                "text": "A short but legally relevant provision.",
            }
        ]
    )

    chunks = build_chunks_for_doc(pages)

    assert len(chunks) == 1
    assert "legally relevant" in str(chunks[0]["chunk_text"])
