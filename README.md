# Banking Regulation Compliance Assistant

**Live demo:** https://compliance-rag.streamlit.app/

## 1. Project Overview

This project is a banking regulation assistant that answers questions using approved Basel and FINMA documents.

That matters because a normal LLM can sound convincing without showing evidence. This system is built for better accountability: it retrieves the right source passages first, answers from those passages, and shows where the answer came from. The result is more useful for compliance work, easier to review, and safer to trust.

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

A  registry in data/metadata/docs.csv defines the regulatory text and its metadata. The pipeline parses the source PDFs, cleans the text, splits it into overlapping paragraph-level chunks, and uses those chunks as the main retrieval units. It then builds two search layers: BM25 for keyword matching and a vector index for semantic search using multilingual embeddings.

When a user asks a question, the system runs both keyword search and embedding-based search, combines the results, and selects the most relevant chunks. Only those retrieved passages are sent to the answer model. This is the main difference from a normal LLM workflow: the answer is based on a limited set of approved source texts instead of relying on the model’s general memory.

The Streamlit app shows the final answer, the supporting excerpts, and a few retrieval controls so the workflow can be inspected. It is still a demo deployment, but it already makes the full process clear and easy to understand.

## 4. Results

What we can clearly show today is that the app is live, works on a bilingual corpus of 22  regulatory documents, and produces evidence-backed answers instead of unsupported free-form responses.

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
