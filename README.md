# Banking Regulation Compliance Assistant

**Live demo:** https://compliance-rag.streamlit.app/

## 1. Project Overview

This project is a retrieval-augmented AI assistant designed to answer banking regulation questions using official Basel and FINMA documents.

Standard AI models can produce fluent answers, but they do not always show where the information comes from. This can be a problem in compliance or regulatory work, where answers must be based on trusted sources. To solve this, the system searches in a curated collection of regulatory documents and generates answers only from those texts.
By grounding each response in official sources, the assistant reduces hallucinations, improves reliability, and makes it possible to verify every statement.
The system also supports both French and English documents, which is important for regulatory research in Switzerland.

Compared with a standard AI model, this assistant is designed to:

-Answer questions using a fixed set of approved regulatory documents

-Show citations from the source text

-Display the retrieved passages for transparency

-Support compliance and audit-friendly workflows

## 2. Problem Definition

Banking regulation is complex, frequently updated, and spread across many official documents such as Basel standards and FINMA guidance.
In practice, compliance analysts often need to answer precise questions about topics like governance responsibilities, liquidity rules, or conduct requirements, while still being able to trace the answer back to the original source.

Mistakes in this process are not only time-consuming but can also create audit issues and increase regulatory risk if decisions cannot be justified with the correct document reference.

To address this, the project proposes a grounded retrieval-augmented workflow.
Approved regulatory documents are stored in a controlled corpus, converted into searchable text, and split into smaller sections. When a question is asked, the system first finds the most relevant passages, then generates an answer based only on those passages, and includes citations to the source.

## 3. Technical Approach

The architecture is intentionally simple. A curated registry in `data/metadata/docs.csv` defines the approved corpus, `src/parse_pdf.py` extracts page text from PDFs, `src/chunk.py` converts that text into paragraph-based overlapping chunks, and `src/index_embeddings.py` builds the FAISS index used for semantic search.

At query time, the system combines BM25 keyword retrieval with multilingual vector retrieval, then sends only the top evidence passages to the answer model. That is the main difference from a normal LLM: the model is not asked to answer from general memory, but from retrieved regulatory text under citation and no-speculation constraints.

The interface is delivered through `app/streamlit_app.py`, which exposes filters, retrieval controls, and source excerpts for inspection. It is a demo-oriented deployment, but it already includes the core mechanisms that make the system more trustworthy than a plain chat wrapper.

## 4. Results

The strongest verified results in the current repository are system-level rather than business KPI claims. The assistant is deployed as a live Streamlit demo, supports a bilingual corpus of 22 curated regulatory documents, and produces citation-grounded answers instead of unsupported free-form responses.

The project also includes a reproducible A/B evaluation workflow in `eval/`:
- `run_ab.py` generates answers for both the RAG pipeline and a non-retrieval baseline
- `score_ab.py` scores keyword recall, refusal accuracy, citation presence, and latency
- `make_report.py` builds a shareable markdown summary

Benchmark output files are not committed in the current repository snapshot, so this README does not claim test-set win rates or business impact numbers that are not backed by stored results. The clearest current value is qualitative: compared with a normal LLM, this system gives users inspectable evidence, narrower answer boundaries, and a workflow that is better aligned with compliance review.

## 5. How to Run

### Prerequisites

- Python 3.12+
- Poetry
- `OPENAI_API_KEY` set in `.env`

### Installation

```powershell
poetry install
```

### Rebuild the Retrieval Pipeline

```powershell
.\.venv\Scripts\python.exe src\metadata.py
.\.venv\Scripts\python.exe src\parse_pdf.py
.\.venv\Scripts\python.exe src\chunk.py
.\.venv\Scripts\python.exe src\index_embeddings.py
```

### Run the App

```powershell
poetry run python -m streamlit run app/streamlit_app.py
```

### Quick Start

After the app launches, ask a question such as:

```text
What governance responsibilities does the board have for internal controls?
```

The app will retrieve supporting passages, generate a grounded answer, and optionally show the underlying excerpts and retrieval scores.

## 6. Project Structure

```text
snb-rag/
|- app/
|  |- streamlit_app.py         # Interactive compliance assistant UI
|- data/
|  |- raw_pdf/                 # Source regulatory PDFs
|  |- metadata/
|  |  |- docs.csv              # Curated document registry
|  |- processed/
|  |  |- pages.parquet         # Parsed page-level text
|  |  |- chunks.parquet        # Retrieval chunks with page spans
|  |- artifacts/
|  |  |- faiss.index           # Vector index
|  |  |- embedding_metadata.parquet
|- eval/
|  |- questions.csv            # Evaluation set
|  |- run_ab.py                # Generate RAG vs baseline answers
|  |- score_ab.py              # Score recall, refusals, citations, latency
|  |- make_report.py           # Build markdown report
|- src/
|  |- metadata.py              # Validate document registry
|  |- parse_pdf.py             # Extract PDF text page by page
|  |- chunk.py                 # Build paragraph-based overlapping chunks
|  |- index_embeddings.py      # Create embeddings and FAISS index
|  |- retrieve_hybrid.py       # Hybrid retrieval logic
|  |- retrieve_vector.py       # Vector retrieval helper
|  |- retrieve.py              # Retrieval entry point
|  |- answer.py                # Grounded answer generation
|- pyproject.toml
|- README.md
```

## 7. Future Improvements

- Add stored benchmark outputs and a stable results table in the README so model-quality claims are reproducible.
- Introduce metadata-aware ranking for time-sensitive queries such as "latest", "current", or version-specific regulatory questions.
- Add a reranker after first-stage retrieval to improve evidence ordering for long or ambiguous questions.
- Make chunking more section-aware so legal structure, headings, and numbered obligations are preserved more explicitly.
- Add monitoring, regression checks, and artifact validation so pipeline failures or retrieval regressions are easier to catch.
- Harden deployment beyond Streamlit demo mode with clearer service boundaries, logging, and operational controls.
