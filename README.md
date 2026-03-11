# Banking Regulation Compliance Assistant (RAG)

**Live demo:** https://compliance-rag.streamlit.app/

## Overview
This project is a retrieval-augmented generation system for banking and regulatory analysis.

It answers questions using a curated corpus of Basel and FINMA documents, retrieves supporting passages, and drafts responses grounded only in those sources.

Current corpus characteristics:
- bilingual corpus: French and English
- manually curated document registry in `data/metadata/docs.csv`
- paragraph-based chunking with overlap
- hybrid retrieval: BM25 + FAISS vector search
- multilingual embeddings for cross-lingual retrieval

## How It Works
End-to-end flow:
1. Register documents manually in `data/metadata/docs.csv`
2. Parse PDFs into page-level text with `src/parse_pdf.py`
3. Build paragraph-based chunks with `src/chunk.py`
4. Create embeddings and FAISS index with `src/index_embeddings.py`
5. Ask questions in `app/streamlit_app.py`
6. Retrieve relevant evidence with hybrid search
7. Generate an answer using only the retrieved context

## Current Architecture
### Document Registry
The source of truth is `data/metadata/docs.csv`.

It stores curated metadata such as:
- `doc_id`
- `title`
- `topic`
- `language`
- `local_path`

`src/metadata.py` validates the registry rather than inferring metadata automatically.

### Parsing
`src/parse_pdf.py` extracts text page by page from each registered PDF and stores the normalized output in `data/processed`.

### Chunking
`src/chunk.py` builds paragraph-oriented chunks with overlap so retrieval works on smaller, denser passages than the original MVP version.

### Retrieval
The app combines:
- BM25 keyword retrieval
- FAISS semantic retrieval
- weighted hybrid ranking
- per-document chunk caps to reduce source domination

### Answer Generation
The app sends retrieved context to OpenAI and instructs the model to:
- use only retrieved evidence
- answer in the same language as the user
- avoid speculation
- cite factual claims with source titles and page ranges

## Tech Stack
- Python
- Pandas, NumPy
- Streamlit
- OpenAI API
- FAISS
- Sentence Transformers
- `rank-bm25`
- Poetry

Current embedding model default:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Current answer model default:
- `gpt-4o-mini`

## Data and Artifacts
Main files:
- `data/metadata/docs.csv`: manual document registry
- `data/processed/chunks.parquet`: retrieval chunks
- `data/artifacts/faiss.index`: vector index
- `data/artifacts/embedding_metadata.parquet`: chunk/index metadata

## Run Locally
### Prerequisites
- Python 3.12+
- Poetry
- `OPENAI_API_KEY` in `.env`

### Install
```powershell
poetry install
```

### Rebuild Pipeline
```powershell
.\.venv\Scripts\python.exe src\metadata.py
.\.venv\Scripts\python.exe src\parse_pdf.py
.\.venv\Scripts\python.exe src\chunk.py
.\.venv\Scripts\python.exe src\index_embeddings.py
```

### Start App
```powershell
poetry run python -m streamlit run app/streamlit_app.py
```

## Project Structure
```text
snb-rag/
|- app/
|  |- streamlit_app.py
|- src/
|  |- metadata.py
|  |- parse_pdf.py
|  |- chunk.py
|  |- index_embeddings.py
|  |- retrieve.py
|  |- retrieve_hybrid.py
|  |- retrieve_vector.py
|  |- answer.py
|- data/
|  |- raw_pdf/
|  |- metadata/
|  |  |- docs.csv
|  |- processed/
|  |- artifacts/
|- eval/
|  |- questions.csv
|  |- run_ab.py
|  |- score_ab.py
|  |- make_report.py
|- pyproject.toml
|- README.md
```

## Current Limitations
- metadata-aware ranking is still limited
- no reranker yet
- chunking is improved but not fully section-aware
- evaluation set still needs to become more realistic and retrieval-labeled
- Streamlit app is suitable for demo/internal usage, not a hardened production deployment boundary

## Recommended Next Improvements
1. add metadata-aware ranking for `latest`, `current`, and version-sensitive queries
2. add a reranker after first-stage hybrid retrieval
3. improve evaluation with realistic questions and gold retrieval labels
4. make chunking section-aware for legal and regulatory structure
5. add monitoring, regression checks, and service hardening
