import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(BASE_DIR / ".env")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

df = pd.read_parquet(CHUNKS_PATH)

model = SentenceTransformer(EMBEDDING_MODEL)

texts = df["chunk_text"].tolist()

print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, str(OUTPUT_DIR / "faiss.index"))

df.to_parquet(OUTPUT_DIR / "embedding_metadata.parquet", index=False)

print(f"Vector index saved with embedding model: {EMBEDDING_MODEL}")
