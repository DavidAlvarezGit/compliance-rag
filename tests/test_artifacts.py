from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.artifacts import load_artifact_manifest


class ArtifactTests(unittest.TestCase):
    def test_load_artifact_manifest_reads_utf8_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(
                json.dumps({"schema_version": 1, "embedding_model": "modèle"}, ensure_ascii=False),
                encoding="utf-8",
            )
            manifest = load_artifact_manifest(manifest_path)
            self.assertEqual(manifest["embedding_model"], "modèle")


if __name__ == "__main__":
    unittest.main()
