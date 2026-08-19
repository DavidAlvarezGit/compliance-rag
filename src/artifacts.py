from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import pandas as pd

MANIFEST_NAME = "index_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(frame, index=False).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def write_manifest(
    path: Path,
    *,
    embedding_model: str,
    dimension: int,
    row_count: int,
    metadata_path: Path,
) -> None:
    payload = {
        "embedding_model": embedding_model,
        "dimension": dimension,
        "row_count": row_count,
        "metadata_sha256": file_sha256(metadata_path),
        "corpus_sha256": dataframe_sha256(pd.read_parquet(metadata_path)),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_artifacts(
    index: faiss.Index,
    metadata: pd.DataFrame,
    *,
    embedding_model: str,
    manifest_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    if not manifest_path.exists():
        raise RuntimeError(
            f"Missing {manifest_path.name}; rebuild embeddings with src/index_embeddings.py."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "embedding_model": embedding_model,
        "dimension": int(index.d),
        "row_count": len(metadata),
        "metadata_sha256": file_sha256(metadata_path),
        "corpus_sha256": dataframe_sha256(metadata),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if int(index.ntotal) != len(metadata):
        mismatches["index_row_count"] = (int(index.ntotal), len(metadata))
    if mismatches:
        details = ", ".join(
            f"{key}: stored={stored!r}, expected={expected_value!r}"
            for key, (stored, expected_value) in mismatches.items()
        )
        raise RuntimeError(f"Retrieval artifacts are inconsistent ({details}). Rebuild the index.")
    return manifest
