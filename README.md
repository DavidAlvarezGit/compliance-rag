# Banking Regulation Compliance Assistant (RAG) — Basel III & FINMA

**Live demo:** (https://compliance-rag.streamlit.app/)

## 1) Project Overview
**What it is**  
An AI assistant that answers questions about banking regulation and the Basel Framework using official regulatory documents (e.g., Basel Committee standards, supervisory guidelines, and regulatory publications) as its knowledge base.

**Why it matters**  
Compliance teams spend significant time verifying that responses are supported by official regulatory text. Answers that are not directly grounded in authoritative documents create audit, compliance, and governance risks.

Unlike a standard ChatGPT-style model that may generate responses without citing a specific source, this system retrieves relevant passages from regulatory documents and bases its answers on them, ensuring traceability and supporting audit requirements.

### Key Results

- **Benchmark evaluation**  
  Assessed on a 40-question A/B benchmark comparing the retrieval-augmented assistant with a standard chat model.

- **Higher answer reliability**  
  Achieved a higher answer-quality score (0.532 vs 0.409), indicating more accurate and policy-grounded responses.

- **Better handling of unsupported queries**  
  Correctly refused questions without supporting evidence 92.5% of the time compared to 72.5% for the standard chat model, reducing the risk of unsupported answers.

- **Interactive prototype**  
  Delivered as a Streamlit application enabling compliance teams to query regulatory texts through a structured Q&A interface.

## 2) Problem Definition
**Business context**  
Banking compliance officers need fast, defensible answers tied to source regulation text. If an answer is not linked to evidence, it is hard to trust and difficult to defend in audits.

**Current solution**  
Most teams either search many PDFs manually or use generic chat tools that may sound correct but are not source-grounded. This increases review time and rework.

**Proposed solution**  
Use a document-aware assistant that first finds relevant policy passages, then writes an answer using only those passages with citations.

**Success metrics**  
We track: answer-quality proxy, refusal accuracy on unsupported questions, citation presence, and response latency.

## 3) Technical Approach
**Data pipeline**  
Regulatory PDFs are split into smaller passages and stored in `data/processed/chunks.parquet`. Document metadata is stored in `data/metadata/docs.csv`.

**Feature engineering**  
The assistant combines keyword search and meaning-based search. This hybrid approach improves retrieval quality while keeping behavior understandable.

**Tech stack used**  
- Python
- Pandas, NumPy
- Streamlit (UI)
- OpenAI API (`gpt-4o-mini`)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS (vector search)
- BM25 via `rank-bm25` (keyword search)
- Poetry (dependency management)

**Model development**  
Baseline is direct chat without document retrieval. The assistant is compared against baseline using `eval/run_ab.py` and scored with `eval/score_ab.py`.

**Deployment**  
The app runs through Streamlit (`app/streamlit_app.py`) with interactive Q&A and evidence inspection.

**Monitoring**  
Quality is tracked with offline A/B evaluations and generated reports (`eval/score_ab.py`, `eval/make_report.py`).

## 4) Results
**Model performance**  
- Overall quality proxy: `RAG 0.532` vs `Baseline 0.409`  
- RAG win rate by question: `35.0%` (ties `20.0%`)  
- Refusal accuracy: `RAG 92.5%` vs `Baseline 72.5%`  
- Average latency: `RAG 6.425s` vs `Baseline 3.166s`

**Business impact**  
The current system is strongest on groundedness and safer behavior (citations + better refusal on unsupported questions), which is aligned with compliance workflows.

**Technical learnings**  
RAG performs best on unsupported/safety scenarios. For fully answerable questions, retrieval quality and context selection remain the biggest improvement lever.

## 5) How to Run
**Prerequisites**  
- Python 3.12+  
- Poetry  
- OpenAI API key in `.env` as `OPENAI_API_KEY`

**Installation**
```powershell
poetry install
```

**Run app**
```powershell
poetry run python -m streamlit run app/streamlit_app.py
```
## 6) Project Structure
```text
snb-rag/
|- app/
|  |- streamlit_app.py          # Compliance-facing app
|- src/
|  |- parse_pdf.py              # PDF parsing
|  |- chunk.py                  # Text chunking
|  |- metadata.py               # Metadata handling
|  |- index_embeddings.py       # Embedding/index utilities
|  |- retrieve_hybrid.py        # Hybrid retrieval
|  |- retrieve.py               # Retrieval helpers
|  |- retrieve_vector.py        # Vector retrieval helper
|  |- answer.py                 # Grounded answer generation
|- data/
|  |- raw_pdf/                  # Source regulatory PDFs
|  |- processed/                # Processed chunks
|  |- metadata/                 # docs.csv and related metadata
|  |- artifacts/                # FAISS index artifacts
|- eval/
|  |- questions.csv             # Benchmark dataset
|  |- run_ab.py                 # Generate A/B outputs
|  |- score_ab.py               # Score benchmark metrics
|  |- make_report.py            # Create markdown report
|  |- report.md                 # Latest benchmark report
|- pyproject.toml
|- README.md
```

## 7) Future Improvements
- Improve answerable-question performance by refining retrieval ranking and context selection
- Add stronger evaluation (LLM-judge rubric + confidence intervals)
- Add automated benchmark checks in CI before merges
- Add role-based report views for compliance committees
- Add scheduled reindexing when new regulatory documents are ingested
