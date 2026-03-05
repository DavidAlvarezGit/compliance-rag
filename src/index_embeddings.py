from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(CHUNKS_PATH)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = df["chunk_text"].tolist()

print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, str(OUTPUT_DIR / "faiss.index"))

df.to_parquet(OUTPUT_DIR / "embedding_metadata.parquet", index=False)

print("Vector index saved.")
