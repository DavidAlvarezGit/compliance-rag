import faiss
import numpy as np
import pandas as pd
import pytest

from src.artifacts import validate_artifacts, write_manifest


def test_artifact_manifest_detects_model_mismatch(tmp_path) -> None:
    metadata = pd.DataFrame({"chunk_text": ["one", "two"]})
    metadata_path = tmp_path / "metadata.parquet"
    manifest_path = tmp_path / "index_manifest.json"
    metadata.to_parquet(metadata_path, index=False)
    index = faiss.IndexFlatIP(3)
    index.add(np.ones((2, 3), dtype="float32"))
    write_manifest(
        manifest_path,
        embedding_model="model-a",
        dimension=3,
        row_count=2,
        metadata_path=metadata_path,
    )

    validate_artifacts(
        index,
        metadata,
        embedding_model="model-a",
        manifest_path=manifest_path,
        metadata_path=metadata_path,
    )
    with pytest.raises(RuntimeError, match="embedding_model"):
        validate_artifacts(
            index,
            metadata,
            embedding_model="model-b",
            manifest_path=manifest_path,
            metadata_path=metadata_path,
        )
