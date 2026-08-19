import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from .artifacts import MANIFEST_NAME, write_manifest
except ImportError:
    from artifacts import MANIFEST_NAME, write_manifest

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "artifacts"
load_dotenv(BASE_DIR / ".env")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

def main() -> None:
    df = pd.read_parquet(CHUNKS_PATH)
    if df.empty:
        raise ValueError("Cannot build an index from an empty chunks file.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Generating embeddings...")
    embeddings = model.encode(
        df["chunk_text"].tolist(), show_progress_bar=True, normalize_embeddings=True
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUT_DIR / "faiss.index"
    metadata_path = OUTPUT_DIR / "embedding_metadata.parquet"
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    temp_index_path = OUTPUT_DIR / "faiss.tmp.index"
    temp_metadata_path = OUTPUT_DIR / "embedding_metadata.tmp.parquet"
    temp_manifest_path = OUTPUT_DIR / "index_manifest.tmp.json"
    faiss.write_index(index, str(temp_index_path))
    df.to_parquet(temp_metadata_path, index=False)
    write_manifest(
        temp_manifest_path,
        embedding_model=EMBEDDING_MODEL,
        dimension=embeddings.shape[1],
        row_count=len(df),
        metadata_path=temp_metadata_path,
    )
    temp_index_path.replace(index_path)
    temp_metadata_path.replace(metadata_path)
    temp_manifest_path.replace(manifest_path)
    print(f"Vector index saved with embedding model: {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
