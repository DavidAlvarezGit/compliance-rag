from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.metadata import load_metadata, validate_metadata


class MetadataTests(unittest.TestCase):
    def test_load_metadata_reads_utf8_titles(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "docs.csv"
            csv_path.write_text(
                "doc_id,doc_type,topic,year,issue,language,local_path,title\n"
                'doc-1,REG_BANK,other,2024,,FR,"D:\\tmp\\règle.pdf","Règles de conduite"\n',
                encoding="utf-8",
            )
            df = load_metadata(csv_path)
            self.assertEqual(df.loc[0, "title"], "Règles de conduite")

    def test_validate_metadata_accepts_matching_pdf_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw_pdf"
            raw_dir.mkdir()
            pdf_path = raw_dir / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")
            df = pd.DataFrame(
                [
                    {
                        "doc_id": "doc-1",
                        "doc_type": "REG_BANK",
                        "topic": "other",
                        "year": 2024,
                        "issue": "",
                        "language": "EN",
                        "local_path": str(pdf_path.resolve()),
                        "title": "Sample",
                    }
                ]
            )
            validate_metadata(df, raw_dir=raw_dir)


if __name__ == "__main__":
    unittest.main()
