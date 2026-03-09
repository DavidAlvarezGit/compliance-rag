import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Fast default model for lower latency.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", str(BASE_DIR / "data" / "processed" / "chunks.parquet"))
DOCS_PATH = os.getenv("DOCS_PATH", str(BASE_DIR / "data" / "metadata" / "docs.csv"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(BASE_DIR / "data" / "artifacts" / "faiss.index"))


@dataclass
class Chunk:
    idx: int
    doc_id: str
    year: int
    page_start: int
    page_end: int
    topic: Optional[str]
    issue: Optional[str]
    chunk_text: str
    bm25: float = 0.0
    vec: float = 0.0
    hybrid: float = 0.0


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if abs(xmax - xmin) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


@st.cache_resource
def load_chunks() -> pd.DataFrame:
    df = pd.read_parquet(CHUNKS_PATH).copy()
    required = {"chunk_text", "doc_id", "year", "page_start", "page_end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"chunks.parquet missing columns: {sorted(missing)}")

    df["year"] = df["year"].apply(lambda v: safe_int(v, 0))
    df["page_start"] = df["page_start"].apply(lambda v: safe_int(v, 0))
    df["page_end"] = df["page_end"].apply(lambda v: safe_int(v, 0))
    if "topic" not in df.columns:
        df["topic"] = None
    if "issue" not in df.columns:
        df["issue"] = None
    return df.reset_index(drop=True)


@st.cache_resource
def load_bm25(chunks_df: pd.DataFrame) -> BM25Okapi:
    tokenized = [str(t).split() for t in chunks_df["chunk_text"].tolist()]
    return BM25Okapi(tokenized)


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_faiss() -> faiss.Index:
    return faiss.read_index(FAISS_INDEX_PATH)


@st.cache_resource
def load_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it in environment or Streamlit secrets.")
    return OpenAI(api_key=api_key)


def load_docs_optional() -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(DOCS_PATH)
    except Exception:
        return None


def retrieve_candidates(
    query: str,
    chunks_df: pd.DataFrame,
    bm25: BM25Okapi,
    embedder: SentenceTransformer,
    index: faiss.Index,
    bm25_k: int,
    vec_k: int,
    w_bm25: float,
    w_vec: float,
) -> List[Chunk]:
    bm25_scores = np.array(bm25.get_scores(query.split()), dtype=float)
    bm25_top_idx = np.argpartition(-bm25_scores, min(bm25_k, len(bm25_scores)) - 1)[:bm25_k]
    bm25_top_idx = bm25_top_idx[np.argsort(-bm25_scores[bm25_top_idx])]

    q_vec = embedder.encode([query], normalize_embeddings=True)
    distances, indices = index.search(q_vec.astype(np.float32), vec_k)
    vec_idx = indices[0].astype(int)
    vec_scores = distances[0].astype(float)
    if getattr(index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_L2:
        vec_scores = -vec_scores

    cand_ids = set(map(int, bm25_top_idx.tolist())) | set(map(int, vec_idx.tolist()))
    cand_ids = [i for i in cand_ids if 0 <= i < len(chunks_df)]

    bm25_cand = bm25_scores[cand_ids]
    bm25_norm = _minmax_norm(bm25_cand)

    vec_map = {int(i): float(s) for i, s in zip(vec_idx, vec_scores) if int(i) >= 0}
    vec_floor = float(np.min(vec_scores)) if vec_scores.size else 0.0
    vec_cand = np.array([vec_map.get(int(i), vec_floor) for i in cand_ids], dtype=float)
    vec_norm = _minmax_norm(vec_cand)

    rows: List[Chunk] = []
    for pos, idx in enumerate(cand_ids):
        row = chunks_df.iloc[idx]
        item = Chunk(
            idx=int(idx),
            doc_id=str(row["doc_id"]),
            year=int(row["year"]),
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            topic=None if pd.isna(row.get("topic", None)) else str(row.get("topic", None)),
            issue=None if pd.isna(row.get("issue", None)) else str(row.get("issue", None)),
            chunk_text=str(row["chunk_text"]),
            bm25=float(bm25_norm[pos]),
            vec=float(vec_norm[pos]),
        )
        item.hybrid = w_bm25 * item.bm25 + w_vec * item.vec
        rows.append(item)

    rows.sort(key=lambda x: x.hybrid, reverse=True)
    return rows


def build_context(chunks: List[Chunk], max_chunks: int) -> str:
    parts = []
    for c in chunks[:max_chunks]:
        parts.append(f"{c.doc_id} pp.{c.page_start}-{c.page_end}:\n{c.chunk_text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    client: OpenAI,
    model: str,
    query: str,
    chunks: List[Chunk],
    temperature: float,
    max_chunks_for_llm: int,
    max_tokens: int,
) -> str:
    context = build_context(chunks, max_chunks=max_chunks_for_llm)
    prompt = f"""
You are a senior banking compliance analyst.
Use only the context below.
If context is insufficient, state this explicitly.
Do not speculate or add outside knowledge.
Write a short compliance brief for compliance officers.
Keep it concise and practical.
Every factual claim must include a citation in this format:
(Source: DOC_ID pp.X-Y)

CONTEXT:
{context}

QUESTION:
{query}
""".strip()

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


st.set_page_config(page_title="Basel III Compliance Assistant", layout="wide")

chunks_df = load_chunks()
bm25 = load_bm25(chunks_df)
embedder = load_embedder()
faiss_index = load_faiss()
docs_df = load_docs_optional()

st.markdown(
    """
<style>
.main .block-container {max-width: 1100px; padding-top: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.2rem;}
</style>
""",
    unsafe_allow_html=True,
)

doc_count = int(chunks_df["doc_id"].nunique())
year_min = int(chunks_df["year"].min()) if len(chunks_df) else 0
year_max = int(chunks_df["year"].max()) if len(chunks_df) else 0

st.title("Basel III Compliance Assistant")
st.caption("Grounded regulatory Q&A for banking compliance and risk stakeholders.")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Chunks", f"{len(chunks_df):,}")
kpi2.metric("Documents", f"{doc_count:,}")
kpi3.metric("Years", f"{year_min} - {year_max}")

tab_ask, tab_inspect, tab_system = st.tabs(["Ask", "Inspect", "System"])

with st.sidebar:
    st.header("Configuration")
    st.subheader("Retrieval Controls")
    bm25_k = st.slider("BM25 Candidate Pool", 10, 200, 20, 10)
    vec_k = st.slider("Vector Candidate Pool", 10, 200, 20, 10)
    w_bm25 = st.slider("BM25 Weight", 0.0, 1.0, 0.5, 0.05)
    w_vec = 1.0 - w_bm25
    st.caption(f"Vector weight: {w_vec:.2f}")

    st.subheader("Model Controls")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", 0.0, 0.5, 0.0, 0.05)
    max_chunks_for_llm = st.slider("Context Chunks to LLM", 2, 16, 5, 1)
    max_tokens = st.slider("Max Completion Tokens", 128, 2000, 500, 50)

    st.subheader("Display Options")
    show_context = st.checkbox("Show Retrieved Context", value=True)
    show_scores = st.checkbox("Show Retrieval Scores", value=False)

with tab_ask:
    st.subheader("Compliance Question")
    st.caption("Example questions for trading/compliance use:")
    ex_col1, ex_col2 = st.columns(2)
    if ex_col1.button(
        "Trading Incident Escalation",
        width="stretch",
        key="example_q1",
    ):
        st.session_state["question_input"] = (
            "How should trading-related operational risk incidents be escalated and managed?"
        )
    if ex_col2.button(
        "Board Internal Controls",
        width="stretch",
        key="example_q2",
    ):
        st.session_state["question_input"] = (
            "What governance responsibilities does the board have for internal controls?"
        )

    with st.form("ask_form", clear_on_submit=False):
        q = st.text_area(
            "Question",
            height=100,
            placeholder="Example: What are the key operational risk resilience obligations in the latest guidance?",
            key="question_input",
        )
        run = st.form_submit_button("Generate Compliance Brief", type="primary", width="stretch")

    if run:
        if not q.strip():
            st.warning("Please enter a compliance or regulatory question.")
        else:
            t0 = time.time()
            with st.spinner("Retrieving relevant source passages..."):
                candidates = retrieve_candidates(
                    query=q.strip(),
                    chunks_df=chunks_df,
                    bm25=bm25,
                    embedder=embedder,
                    index=faiss_index,
                    bm25_k=int(bm25_k),
                    vec_k=int(vec_k),
                    w_bm25=float(w_bm25),
                    w_vec=float(w_vec),
                )

            st.success(f"Retrieved {len(candidates)} source passages in {time.time() - t0:.2f}s")

            if show_context:
                with st.expander("Retrieved Evidence", expanded=False):
                    for i, c in enumerate(candidates[:max_chunks_for_llm], start=1):
                        st.markdown(f"**{i}. {c.doc_id} | {c.year} | pp.{c.page_start}-{c.page_end}**")
                        if show_scores:
                            st.caption(f"hybrid={c.hybrid:.4f} | bm25={c.bm25:.3f} | vec={c.vec:.3f}")
                        st.write(c.chunk_text)
                        st.divider()

            with st.spinner("Drafting grounded compliance response..."):
                client = load_openai_client()
                answer = generate_answer(
                    client=client,
                    model=model.strip(),
                    query=q.strip(),
                    chunks=candidates,
                    temperature=float(temperature),
                    max_chunks_for_llm=int(max_chunks_for_llm),
                    max_tokens=int(max_tokens),
                )

            st.markdown("### Compliance Response")
            st.write(answer)

with tab_inspect:
    st.subheader("Evidence Inspector")
    q2 = st.text_input("Query for Inspection", value="")
    top_n = st.slider("Rows to Display", 10, 120, 40, 10)
    if st.button("Run Evidence Inspection", width="content") and q2.strip():
        candidates = retrieve_candidates(
            query=q2.strip(),
            chunks_df=chunks_df,
            bm25=bm25,
            embedder=embedder,
            index=faiss_index,
            bm25_k=int(bm25_k),
            vec_k=int(vec_k),
            w_bm25=float(w_bm25),
            w_vec=float(w_vec),
        )

        df = pd.DataFrame(
            [
                {
                    "rank": rank,
                    "doc_id": c.doc_id,
                    "year": c.year,
                    "topic": c.topic,
                    "issue": c.issue,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "bm25": c.bm25,
                    "vec": c.vec,
                    "hybrid": c.hybrid,
                    "preview": (c.chunk_text[:220] + "...") if len(c.chunk_text) > 220 else c.chunk_text,
                }
                for rank, c in enumerate(candidates[:top_n], start=1)
            ]
        )
        st.dataframe(df, width="stretch", height=460)

with tab_system:
    st.subheader("Methodology Overview")
    st.markdown(
        """
- Retrieval: hybrid ranking (BM25 + FAISS)
- Answer generation: grounded on retrieved evidence only
- Scope: streamlined workflow with no recency reranking and no metadata boosts
        """.strip()
    )

    st.markdown("**Runtime Configuration**")
    cfg = pd.DataFrame(
        [
            {"parameter": "model", "value": model},
            {"parameter": "bm25_k", "value": int(bm25_k)},
            {"parameter": "vec_k", "value": int(vec_k)},
            {"parameter": "bm25_weight", "value": float(w_bm25)},
            {"parameter": "vector_weight", "value": float(w_vec)},
            {"parameter": "temperature", "value": float(temperature)},
            {"parameter": "max_chunks_for_llm", "value": int(max_chunks_for_llm)},
            {"parameter": "max_tokens", "value": int(max_tokens)},
        ]
    )
    cfg["value"] = cfg["value"].astype(str)
    st.dataframe(cfg, width="stretch", hide_index=True)

    if docs_df is not None:
        with st.expander("Document Register (docs.csv)", expanded=False):
            st.dataframe(docs_df, width="stretch", height=350)
    else:
        st.info("Document register (docs.csv) not found.")

    st.caption(f"Loaded passages: {len(chunks_df):,} | Index type: {type(faiss_index).__name__}")
