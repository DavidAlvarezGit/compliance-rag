# Banking Regulation Compliance Assistant

**Live demo:** https://compliance-rag.streamlit.app/

## 1. Project Overview

This project is a retrieval-augmented generation system that answers banking regulation questions using a curated corpus of Basel and FINMA source documents.

It matters because a normal LLM can produce fluent answers without proving where they came from, which is a poor fit for compliance work. This system adds value by grounding every answer in approved source documents, reducing hallucination risk, preserving traceability, and supporting bilingual regulatory research across French and English materials.

Current headline facts:
- 22 manually curated regulatory documents in the source registry
- bilingual corpus across French and English
- hybrid retrieval with BM25 plus FAISS vector search
- live Streamlit demo for interactive testing
- evaluation harness for RAG vs non-RAG A/B comparison

Compared with a normal LLM, the assistant is designed to:
- answer from a fixed regulatory corpus instead of open-ended model memory
- cite supporting passages for factual claims
- refuse or narrow answers when evidence is missing
- make retrieved evidence visible to the user

## 2. Problem Definition

Banking and capital-markets regulation is dense, versioned, and spread across multiple supervisory texts. A compliance analyst or reviewer often needs to answer targeted questions such as governance responsibilities, liquidity obligations, or conduct requirements without losing traceability to the original source. The cost of getting this wrong is not just wasted analyst time; it is weak auditability and increased regulatory risk.

The current manual workflow is usually document search, page scanning, and summarization by hand. That works for narrow questions, but it does not scale well across a bilingual corpus, and it makes consistency difficult when different users phrase similar questions differently.

The proposed solution is a grounded RAG workflow: register approved source documents, parse PDFs into page-level text, convert pages into denser paragraph-based chunks, retrieve the strongest evidence with hybrid search, and generate an answer that must stay within the retrieved context and include citations. Success is measured here by retrieval quality, grounded answer quality, citation presence, refusal behavior when evidence is insufficient, and practical usability through a deployable demo.

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
