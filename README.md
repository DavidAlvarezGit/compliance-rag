# Banking Regulation Compliance Assistant

**Live demo:** https://compliance-rag.streamlit.app/

## 1. Project Overview

This project is a banking regulation assistant that answers questions using approved Basel and FINMA documents, not open-ended model memory.

That matters because a normal LLM can sound convincing without showing evidence. This system is built for a higher bar: it retrieves the right source passages first, answers from those passages, and shows where the answer came from. The result is more useful for compliance work, easier to review, and safer to trust.

Compared with a normal LLM, the assistant is designed to:
- answer from a fixed regulatory corpus
- cite the evidence behind factual claims
- stay narrow when the sources are weak
- let the user inspect the supporting text

## 2. Problem Definition

Banking regulation is long, fragmented, and difficult to search quickly. Teams often need answers about governance, liquidity, conduct, or capital rules, but they also need to know exactly which source supports the answer.

The usual alternative is manual search or a general-purpose chatbot. Manual review is slow. A plain LLM is faster, but it may answer beyond the source material and make review harder.

The solution here is a grounded RAG workflow: curate the document set, retrieve the best evidence, and generate an answer that stays tied to that evidence. Success is measured by answer quality, citation quality, refusal behavior when evidence is missing, and whether the tool is practical to use in a live demo.

## 3. Technical Approach

The architecture is intentionally simple. A curated registry in `data/metadata/docs.csv` defines the approved sources. The pipeline parses PDFs, turns them into paragraph-based chunks, and builds both keyword and embedding-based search indexes.

At question time, the system combines keyword search with embedding-based semantic search, selects the most relevant passages, and sends only that evidence to the answer model. This is the key difference from a normal LLM: the answer is built from retrieved regulatory text, not from open-ended model memory.

The Streamlit app exposes the answer, the retrieved excerpts, and a few controls for inspection. It is a demo deployment, but it already shows the workflow clearly and credibly.

## 4. Results

What we can clearly show today is that the assistant is live, works on a bilingual corpus of 22 curated regulatory documents, and produces evidence-backed answers instead of unsupported free-form responses.

The project also includes a reproducible A/B evaluation workflow in `eval/`:
- `run_ab.py` generates answers for both the RAG pipeline and a non-retrieval baseline
- `score_ab.py` scores keyword recall, refusal accuracy, citation presence, and latency
- `make_report.py` builds a shareable markdown summary

Benchmark output files are not committed in the current repository snapshot, so this README avoids unsupported performance claims. The practical value is straightforward: compared with a normal LLM, this system gives users inspectable evidence, tighter answer boundaries, and a workflow that fits compliance review better.

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

The app will retrieve supporting passages, draft an answer from those passages, and show the underlying evidence.

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

- Add stored benchmark outputs so README claims are reproducible.
- Improve ranking for time-sensitive or version-specific queries.
- Add a reranker to improve evidence ordering.
- Make chunking more aware of legal sections and headings.
- Add monitoring and regression checks.
- Harden deployment beyond Streamlit demo mode.
