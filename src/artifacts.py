from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"
ARTIFACT_DIR = BASE_DIR / "data" / "artifacts"
EMBEDDING_METADATA_PATH = ARTIFACT_DIR / "embedding_metadata.parquet"
FAISS_INDEX_PATH = ARTIFACT_DIR / "faiss.index"
ARTIFACT_MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"
ARTIFACT_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_artifact_manifest(
    *,
    chunks_path: Path = CHUNKS_PATH,
    embedding_metadata_path: Path = EMBEDDING_METADATA_PATH,
    faiss_index_path: Path = FAISS_INDEX_PATH,
    embedding_model: str,
) -> dict[str, Any]:
    chunks_df = pd.read_parquet(chunks_path)
    embedding_df = pd.read_parquet(embedding_metadata_path)
    index = faiss.read_index(str(faiss_index_path))
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "embedding_model": embedding_model,
        "chunks_path": str(chunks_path),
        "embedding_metadata_path": str(embedding_metadata_path),
        "faiss_index_path": str(faiss_index_path),
        "chunks_rows": int(len(chunks_df)),
        "embedding_rows": int(len(embedding_df)),
        "faiss_ntotal": int(index.ntotal),
        "embedding_dimension": int(index.d),
        "faiss_metric_type": int(getattr(index, "metric_type", 0)),
        "chunks_sha256": file_sha256(chunks_path),
        "embedding_metadata_sha256": file_sha256(embedding_metadata_path),
        "faiss_index_sha256": file_sha256(faiss_index_path),
    }


def write_artifact_manifest(
    manifest: dict[str, Any],
    manifest_path: Path = ARTIFACT_MANIFEST_PATH,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_artifact_manifest(
    manifest_path: Path = ARTIFACT_MANIFEST_PATH,
) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Artifact manifest not found: {manifest_path}. Rebuild embeddings to create it."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_artifact_manifest(
    *,
    chunks_path: Path = CHUNKS_PATH,
    embedding_metadata_path: Path = EMBEDDING_METADATA_PATH,
    faiss_index_path: Path = FAISS_INDEX_PATH,
    manifest_path: Path = ARTIFACT_MANIFEST_PATH,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    manifest = load_artifact_manifest(manifest_path)
    expected_schema = int(manifest.get("schema_version", -1))
    if expected_schema != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Artifact manifest schema mismatch: expected {ARTIFACT_SCHEMA_VERSION}, got {expected_schema}"
        )

    chunks_df = pd.read_parquet(chunks_path)
    embedding_df = pd.read_parquet(embedding_metadata_path)
    index = faiss.read_index(str(faiss_index_path))
    checks = {
        "chunks_rows": int(len(chunks_df)),
        "embedding_rows": int(len(embedding_df)),
        "faiss_ntotal": int(index.ntotal),
        "embedding_dimension": int(index.d),
        "faiss_metric_type": int(getattr(index, "metric_type", 0)),
        "chunks_sha256": file_sha256(chunks_path),
        "embedding_metadata_sha256": file_sha256(embedding_metadata_path),
        "faiss_index_sha256": file_sha256(faiss_index_path),
    }
    if embedding_model is not None:
        checks["embedding_model"] = embedding_model
    for key, actual in checks.items():
        expected = manifest.get(key)
        if expected != actual:
            raise RuntimeError(
                f"Artifact manifest mismatch for {key}: expected {expected!r}, got {actual!r}"
            )
    return manifest
