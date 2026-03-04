# app.py
# Streamlit UI for your SNB RAG Intelligence System (hybrid retrieval + time/metadata logic + diversity + grounded LLM).
# Assumes you already have:
# - data/processed/chunks.parquet  (must include: text, doc_id, year, issue, page_start, page_end)
# - faiss_index/index.faiss
# - faiss_index/embeddings.npy     (optional; only needed for sanity checks, FAISS index is enough)
# - data/metadata/docs.csv         (optional, for display)

import os
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# OpenAI (new-style client)
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
# =========================
# Config
# =========================
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", str(BASE_DIR / "data" / "processed" / "chunks.parquet"))
DOCS_PATH = os.getenv("DOCS_PATH", str(BASE_DIR / "data" / "metadata" / "docs.csv"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(BASE_DIR / "data" / "artifacts" / "faiss.index"))

REL_TIME_KEYWORDS = [
    "récent", "récente", "récemment", "actuel", "actuelle", "dernière", "dernier",
    "aujourd", "évolution récente", "ces derniers", "ces dernières", "depuis"
]


# =========================
# Data structures
# =========================
@dataclass
class Chunk:
    idx: int
    doc_id: str
    year: int
    issue: Optional[str]
    page_start: int
    page_end: int
    chunk_text: str
    bm25: float = 0.0
    vec: float = 0.0
    hybrid: float = 0.0
    boosts: float = 0.0


# =========================
# Utilities
# =========================
def _minmax_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if abs(xmax - xmin) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def extract_explicit_year(q: str) -> Optional[int]:
    m = re.search(r"\b(20\d{2})\b", q)
    return int(m.group(1)) if m else None


def has_relative_time(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in REL_TIME_KEYWORDS)


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# =========================
# Loaders (cached)
# =========================
@st.cache_resource
def load_chunks() -> pd.DataFrame:
    df = pd.read_parquet(CHUNKS_PATH)

    required = {"chunk_text", "doc_id", "year", "page_start", "page_end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"chunks.parquet missing columns: {sorted(missing)}")

    # Ensure types
    df = df.copy()
    df["year"] = df["year"].apply(lambda v: safe_int(v, 0))
    df["page_start"] = df["page_start"].apply(lambda v: safe_int(v, 0))
    df["page_end"] = df["page_end"].apply(lambda v: safe_int(v, 0))
    if "issue" not in df.columns:
        df["issue"] = None

    # Ensure stable integer index mapping for FAISS ids and BM25 list positions
    df = df.reset_index(drop=True)
    return df


@st.cache_resource
def load_bm25(chunks_df: pd.DataFrame) -> BM25Okapi:
    # Simple whitespace tokenization; keep consistent with your pipeline
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
    return OpenAI()


def load_docs_optional() -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(DOCS_PATH)
    except Exception:
        return None


# =========================
# Retrieval
# =========================
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
    texts = chunks_df["chunk_text"].tolist()

    # BM25 scores across all docs
    q_tokens = query.split()
    bm25_scores = np.array(bm25.get_scores(q_tokens), dtype=float)

    # Take top bm25_k by score
    bm25_top_idx = np.argpartition(-bm25_scores, min(bm25_k, len(bm25_scores)) - 1)[:bm25_k]
    bm25_top_idx = bm25_top_idx[np.argsort(-bm25_scores[bm25_top_idx])]

    # Vector retrieval from FAISS
    q_vec = embedder.encode([query], normalize_embeddings=True)
    # NOTE: assumes your FAISS index was built on normalized embeddings or an appropriate metric.
    D, I = index.search(q_vec.astype(np.float32), vec_k)
    vec_idx = I[0].astype(int)
    vec_scores = D[0].astype(float)

    # Union candidate set
    cand_ids = set(map(int, bm25_top_idx.tolist())) | set(map(int, vec_idx.tolist()))
    cand_ids = [i for i in cand_ids if 0 <= i < len(chunks_df)]

    # Normalize scores over candidate set (not whole corpus) for stable mixing
    bm25_cand = bm25_scores[cand_ids]
    bm25_cand_norm = _minmax_norm(bm25_cand)

    # Vector scores: FAISS distances/sims are only for vec-retrieved ids; others get min
    vec_map: Dict[int, float] = {int(i): float(s) for i, s in zip(vec_idx, vec_scores) if int(i) >= 0}
    vec_cand = np.array([vec_map.get(int(i), float(np.min(vec_scores)) if vec_scores.size else 0.0) for i in cand_ids], dtype=float)
    vec_cand_norm = _minmax_norm(vec_cand)

    # Build Chunk objects
    chunks: List[Chunk] = []
    for pos, idx in enumerate(cand_ids):
        row = chunks_df.iloc[idx]
        c = Chunk(
    idx=int(idx),
    doc_id=str(row["doc_id"]),
    year=int(row["year"]),
    issue=None if pd.isna(row.get("issue", None)) else str(row.get("issue", None)),
    page_start=int(row["page_start"]),
    page_end=int(row["page_end"]),
    chunk_text=str(row["chunk_text"]),
    bm25=float(bm25_cand_norm[pos]),
    vec=float(vec_cand_norm[pos]),
)
        c.hybrid = w_bm25 * c.bm25 + w_vec * c.vec
        chunks.append(c)

    chunks.sort(key=lambda x: x.hybrid, reverse=True)
    return chunks


# =========================
# Time handling + boosts + diversity
# =========================
def apply_time_logic(query: str, cands: List[Chunk], recency_boost: float) -> List[Chunk]:
    explicit_year = extract_explicit_year(query)
    if explicit_year is not None:
        return [c for c in cands if c.year == explicit_year]

    if has_relative_time(query):
        years = [c.year for c in cands if c.year > 0]
        if not years:
            return cands
        mn, mx = min(years), max(years)
        denom = (mx - mn) if (mx - mn) != 0 else 1
        for c in cands:
            if c.year > 0:
                w = (c.year - mn) / denom
                c.boosts += recency_boost * w
                c.hybrid += recency_boost * w
        cands.sort(key=lambda x: x.hybrid, reverse=True)
    return cands


def apply_metadata_boosts(query: str, cands: List[Chunk], boosts: Dict[str, float]) -> List[Chunk]:
    ql = query.lower()

    for c in cands:
        tl = c.chunk_text.lower()

        # simple, deterministic, metadata-aware nudges
        if "inflation" in ql and "inflation" in tl:
            c.boosts += boosts["inflation_match"]
        if "risque" in ql and "risque" in tl:
            c.boosts += boosts["risk_match"]
        if ("politique monétaire" in ql) or ("taux directeur" in ql):
            # early pages often have policy discussion (heuristic)
            if c.page_start <= boosts["policy_page_threshold"]:
                c.boosts += boosts["policy_early_pages"]

        c.hybrid += c.boosts

    cands.sort(key=lambda x: x.hybrid, reverse=True)
    return cands


def apply_diversity(cands: List[Chunk], max_per_doc: int) -> List[Chunk]:
    selected: List[Chunk] = []
    counts: Dict[str, int] = {}
    for c in cands:
        k = c.doc_id
        if counts.get(k, 0) < max_per_doc:
            selected.append(c)
            counts[k] = counts.get(k, 0) + 1
    return selected


# =========================
# LLM answering
# =========================
def build_context(chunks: List[Chunk], max_chunks: int) -> str:
    ctx_parts = []
    for c in chunks[:max_chunks]:
        ctx_parts.append(
            f"{c.doc_id} pp.{c.page_start}-{c.page_end}:\n{c.chunk_text}"
        )
    return "\n\n---\n\n".join(ctx_parts)


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
Tu es un assistant d'analyse macroéconomique spécialisé SNB.

Règles strictes:
- Utilise UNIQUEMENT le CONTEXTE fourni.
- Si le contexte est insuffisant, dis-le explicitement et n'invente rien.
- Toute affirmation factuelle doit être suivie d'une citation au format EXACT:
  (Source: DOC_ID pp.X-Y)

CONTEXTE:
{context}

QUESTION:
{query}

Format de sortie:
RÉPONSE SYNTHÉTIQUE:
- (3 à 6 puces, chacune avec citation)

ANALYSE DÉTAILLÉE:
- (2 à 6 paragraphes, citations fréquentes)
""".strip()

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="SNB RAG Intelligence", layout="wide")

chunks_df = load_chunks()
bm25 = load_bm25(chunks_df)
embedder = load_embedder()
faiss_index = load_faiss()
docs_df = load_docs_optional()

st.title("SNB RAG Intelligence System")
st.caption("Hybrid retrieval (BM25 + FAISS) • time-aware reranking • metadata boosts • diversity control • grounded answers")

tab_ask, tab_inspect, tab_about = st.tabs(["Ask", "Retrieval Inspector", "System"])

with st.sidebar:
    st.header("Retrieval")
    bm25_k = st.slider("BM25 candidates", 10, 200, 40, 10)
    vec_k = st.slider("Vector candidates", 10, 200, 40, 10)
    w_bm25 = st.slider("Weight BM25", 0.0, 1.0, 0.5, 0.05)
    w_vec = 1.0 - w_bm25
    st.caption(f"Weight Vector = {w_vec:.2f}")

    st.header("Time handling")
    recency_boost = st.slider("Recency boost (relative time)", 0.0, 1.0, 0.35, 0.05)

    st.header("Metadata boosts")
    inflation_match = st.slider("Inflation match boost", 0.0, 0.5, 0.15, 0.01)
    risk_match = st.slider("Risk match boost", 0.0, 0.5, 0.15, 0.01)
    policy_early_pages = st.slider("Policy early-page boost", 0.0, 0.5, 0.10, 0.01)
    policy_page_threshold = st.slider("Policy page threshold", 1, 60, 20, 1)

    st.header("Diversity")
    max_per_doc = st.slider("Max chunks per doc_id", 1, 6, 2, 1)

    st.header("LLM")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", 0.0, 0.5, 0.0, 0.05)
    max_chunks_for_llm = st.slider("Chunks passed to LLM", 2, 16, 8, 1)
    max_tokens = st.slider("Max completion tokens", 128, 2000, 900, 50)

    show_context = st.checkbox("Show retrieved context", value=True)
    show_scores = st.checkbox("Show scoring details", value=False)

boost_params = {
    "inflation_match": float(inflation_match),
    "risk_match": float(risk_match),
    "policy_early_pages": float(policy_early_pages),
    "policy_page_threshold": int(policy_page_threshold),
}

with tab_ask:
    q = st.text_area("Question", height=90, placeholder="Ex: Quelles sont les évolutions récentes de l'inflation en Suisse ?")
    col_a, col_b = st.columns([1, 2])
    run = col_a.button("Run", type="primary", use_container_width=True)

    if run and q.strip():
        t0 = time.time()

        with st.status("Retrieving…", expanded=False):
            cands = retrieve_candidates(
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
            cands = apply_time_logic(q, cands, recency_boost=float(recency_boost))
            cands = apply_metadata_boosts(q, cands, boosts=boost_params)
            selected = apply_diversity(cands, max_per_doc=int(max_per_doc))

        st.success(f"Selected {len(selected)} chunks • retrieval time: {time.time() - t0:.2f}s")

        if show_context:
            with st.expander("Retrieved context (top selected)", expanded=False):
                for i, c in enumerate(selected[:max_chunks_for_llm], start=1):
                    header = f"{i}. {c.doc_id} | {c.year} | pp.{c.page_start}-{c.page_end}"
                    st.markdown(f"**{header}**")
                    if show_scores:
                        st.caption(f"hybrid={c.hybrid:.4f} | bm25={c.bm25:.3f} | vec={c.vec:.3f} | boosts={c.boosts:.3f}")
                    st.write(c.chunk_text)
                    st.divider()

        with st.status("Generating answer…", expanded=False):
            client = load_openai_client()
            answer = generate_answer(
                client=client,
                model=model.strip(),
                query=q.strip(),
                chunks=selected,
                temperature=float(temperature),
                max_chunks_for_llm=int(max_chunks_for_llm),
                max_tokens=int(max_tokens),
            )

        st.markdown("### Output")
        st.write(answer)

with tab_inspect:
    st.subheader("Retrieval Inspector")
    q2 = st.text_input("Query to inspect", value="")
    if st.button("Inspect", use_container_width=False) and q2.strip():
        cands = retrieve_candidates(
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
        cands = apply_time_logic(q2, cands, recency_boost=float(recency_boost))
        cands = apply_metadata_boosts(q2, cands, boosts=boost_params)

        df = pd.DataFrame([{
            "rank": r,
            "doc_id": c.doc_id,
            "year": c.year,
            "issue": c.issue,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "bm25": c.bm25,
            "vec": c.vec,
            "boosts": c.boosts,
            "hybrid": c.hybrid,
            "preview": (c.chunk_text[:220] + "…") if len(c.chunk_text) > 220 else c.chunk_text
        } for r, c in enumerate(cands[:120], start=1)])

        st.dataframe(df, use_container_width=True, height=520)

        st.caption("Tip: enable “Show scoring details” in the sidebar to see per-chunk scores in the Ask tab.")

with tab_about:
    st.subheader("System")
    st.markdown(
        """
- **Corpus**: SNB Quarterly Bulletins (FR), 2020–2025  
- **Retrieval**: BM25 + FAISS, score mixing + reranking  
- **Time-aware**: explicit year filter; relative-time recency rerank  
- **Metadata-aware**: heuristic boosts (inflation/risk/policy)  
- **Diversity**: max chunks per doc_id  
- **Answering**: grounded LLM with mandatory citations (doc_id + page range)
        """.strip()
    )

    if docs_df is not None:
        with st.expander("Docs metadata (docs.csv)", expanded=False):
            st.dataframe(docs_df, use_container_width=True, height=350)
    else:
        st.info("docs.csv not found (optional).")

    st.caption(
        f"Loaded chunks: {len(chunks_df):,} • FAISS index: {type(faiss_index).__name__} • Model: {DEFAULT_MODEL}"
    )
