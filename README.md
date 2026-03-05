# Basel III Compliance Assistant (RAG)

## 1) Project Overview
**What it is**  
A retrieval-augmented generation (RAG) system that answers Basel III and related banking regulation questions using only curated regulatory documents.

**Why it matters**  
Compliance teams lose time validating whether generated answers are grounded in actual policy text. Ungrounded responses create audit and governance risk.

**Key results**  
- Evaluated on a 40-question A/B benchmark (`RAG` vs `normal chat without retrieval`)  
- Higher overall keyword-recall proxy: `0.532` vs `0.409`  
- Stronger refusal behavior on unanswerable questions: `92.5%` vs `72.5%`  
- Citation presence in RAG responses: `65.0%`  
- Delivered as a Streamlit app for interactive compliance Q&A and evidence inspection

## 2) Problem Definition
**Business context**  
Banking compliance officers need fast, defensible answers tied to source regulation text. A response without evidence is hard to trust and difficult to defend in internal reviews or audits.

**Current solution**  
Typical workflow is manual lookup across multiple PDF documents or use of generic chat tools that can produce plausible but ungrounded responses. This increases review effort and rework.

**Proposed solution**  
Use hybrid retrieval (BM25 + vector search) over a curated regulation corpus, then generate answers constrained to retrieved evidence and citation format.

**Success metrics**  
Success is measured by answer quality and safety: keyword-recall proxy, refusal accuracy on unanswerable questions, citation presence, and response latency.

## 3) Technical Approach
**Data pipeline**  
Regulatory PDFs are parsed, chunked, and stored in parquet format (`data/processed/chunks.parquet`). Metadata is maintained in `data/metadata/docs.csv`. FAISS index artifacts are stored in `data/artifacts/`.

Quality challenges include heterogeneous document structure and uneven section granularity. The pipeline normalizes chunk schema (`doc_id`, `year`, `page_start`, `page_end`, `chunk_text`) so retrieval remains consistent.

**Feature engineering**  
The current version intentionally stays simple: lexical relevance (BM25) + semantic relevance (MiniLM embeddings + FAISS). Hybrid score is a weighted combination of normalized BM25 and vector scores.

This design was chosen for interpretability and maintainability for a compliance use case, where deterministic retrieval behavior is preferred over heavy heuristic layering.

**Model development**  
Baseline is direct chat completion without retrieval context. The RAG system is compared against this baseline in `eval/run_ab.py`.

The answering model defaults to `gpt-4o-mini` for speed/cost balance, with responses forced to cite source spans. Evaluation scripts compute comparative metrics and generate a markdown report.

**Deployment**  
The system is served as a Streamlit app (`app/streamlit_app.py`) for internal use. It supports question answering, retrieval inspection, and system/runtime configuration visibility.

Serving pattern is interactive request/response; retrieval and index objects are cached in memory to reduce repeated initialization overhead.

**Monitoring**  
Current monitoring is evaluation-driven via offline A/B runs (`eval/score_ab.py` + `eval/make_report.py`). Core tracked metrics: recall proxy, refusal behavior, citation presence, latency.

Retraining/reindexing strategy is manual: update corpus, regenerate chunks/index, rerun benchmark, and compare against previous report before release.

## 4) Results
**Model performance**  
- Overall recall proxy: `RAG 0.532` vs `Baseline 0.409`  
- RAG win rate by question: `35.0%` (ties `20.0%`)  
- Refusal accuracy: `RAG 92.5%` vs `Baseline 72.5%`  
- Average latency: `RAG 6.425s` vs `Baseline 3.166s`

**Business impact**  
The current system is strongest on groundedness and safety behavior (citations + better refusal on unsupported queries), which is aligned with compliance workflows.

**Technical learnings**  
Detailed split analysis shows a key tradeoff: RAG performs better on unanswerable/safety scenarios, while baseline can outperform on some answerable questions. This highlights retrieval quality and context selection as the highest-leverage improvement area.

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

**Run benchmark (A/B)**
```powershell
$env:PYTHONPATH='.'
poetry run python eval/run_ab.py
poetry run python eval/score_ab.py
poetry run python eval/make_report.py
```

## 6) Project Structure
```text
snb-rag/
├── app/
│   └── streamlit_app.py          # Compliance-facing Streamlit interface
├── src/
│   ├── parse_pdf.py              # PDF parsing utilities
│   ├── chunk.py                  # Chunking pipeline
│   ├── metadata.py               # Metadata generation/handling
│   ├── index_embeddings.py       # Embedding + FAISS index build helpers
│   ├── retrieve_hybrid.py        # Core BM25 + FAISS hybrid retrieval
│   ├── retrieve.py               # Retrieval helpers
│   ├── retrieve_vector.py        # Vector-only retrieval helper
│   └── answer.py                 # Grounded answer generation
├── data/
│   ├── raw_pdf/                  # Source regulatory PDFs
│   ├── processed/                # Processed chunks/features
│   ├── metadata/                 # docs.csv and related metadata
│   └── artifacts/                # FAISS index and retrieval artifacts
├── eval/
│   ├── questions.csv             # Benchmark dataset
│   ├── run_ab.py                 # Generate RAG vs baseline outputs
│   ├── score_ab.py               # Score benchmark metrics
│   ├── make_report.py            # Produce markdown evaluation report
│   └── report.md                 # Latest benchmark report
├── pyproject.toml
└── README.md
```

## 7) Future Improvements
- Improve answerable-question performance by refining retrieval ranking and context selection strategy
- Add a stronger evaluator (LLM-as-judge rubric + confidence intervals) for more rigorous claims
- Add automated regression checks in CI for benchmark drift across commits
- Add role-based views and export-ready briefing format for compliance committees
- Add scheduled reindexing pipeline when new regulatory documents are ingested
