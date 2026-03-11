from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
CHUNKS_PATH = os.getenv("CHUNKS_PATH", str(BASE_DIR / "data" / "processed" / "chunks.parquet"))
DOCS_PATH = os.getenv("DOCS_PATH", str(BASE_DIR / "data" / "metadata" / "docs.csv"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(BASE_DIR / "data" / "artifacts" / "faiss.index"))

TOPIC_LABELS = {
    "capital_requirements_framework": "Capital Requirements",
    "corporate_governance_internal_controls": "Governance and Controls",
    "liquidity_risk_management": "Liquidity Risk",
    "climate_nature_related_financial_risks": "Climate and Nature Risk",
    "operational_risk_framework": "Operational Risk",
    "market_conduct_rules": "Market Conduct",
    "credit_risk_standardized_approach": "Credit Risk",
    "irb_framework": "IRB",
    "liquidity_coverage_ratio_lcr": "LCR",
    "net_stable_funding_ratio_nsfr": "NSFR",
    "leverage_ratio_rules": "Leverage Ratio",
    "other": "Other",
}

EXAMPLE_QUESTIONS = [
    "What governance responsibilities does the board have for internal controls?",
    "How should trading-related operational risk incidents be escalated and managed?",
    "Quelles sont les obligations principales en matiere de risque de liquidite ?",
    "What does the current corpus say about climate and nature-related financial risk governance?",
]


@dataclass
class Chunk:
    idx: int
    doc_id: str
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


def normalize_query_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"[a-z0-9]+", normalized)


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if abs(xmax - xmin) < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


@st.cache_data
def load_docs() -> pd.DataFrame:
    df = pd.read_csv(DOCS_PATH).copy()
    expected = {"doc_id", "title", "topic", "language"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"docs.csv missing columns: {sorted(missing)}")
    return df


@st.cache_resource
def load_chunks() -> pd.DataFrame:
    df = pd.read_parquet(CHUNKS_PATH).copy()
    required = {"chunk_text", "doc_id", "page_start", "page_end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"chunks.parquet missing columns: {sorted(missing)}")
    df["page_start"] = df["page_start"].apply(lambda v: safe_int(v, 0))
    df["page_end"] = df["page_end"].apply(lambda v: safe_int(v, 0))
    if "topic" not in df.columns:
        df["topic"] = None
    if "issue" not in df.columns:
        df["issue"] = None
    return df.reset_index(drop=True)


@st.cache_resource
def load_bm25(chunks_df: pd.DataFrame) -> BM25Okapi:
    tokenized = [normalize_query_tokens(text) for text in chunks_df["chunk_text"].tolist()]
    return BM25Okapi(tokenized)


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


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


def build_doc_lookup(docs_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in docs_df.itertuples(index=False):
        lookup[str(row.doc_id)] = {
            "title": str(row.title),
            "topic": str(row.topic),
            "language": str(row.language),
        }
    return lookup


def source_title(doc_lookup: dict[str, dict[str, str]], doc_id: str, topic: Optional[str]) -> str:
    if doc_id in doc_lookup:
        title = doc_lookup[doc_id].get("title", "").strip()
        if title:
            return title
    if topic:
        return TOPIC_LABELS.get(topic, topic.replace("_", " ").title())
    return doc_id


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
    allowed_topics: set[str],
    allowed_languages: set[str],
    max_chunks_per_doc: int,
) -> list[Chunk]:
    candidate_df = chunks_df
    if allowed_topics:
        candidate_df = candidate_df[candidate_df["topic"].isin(allowed_topics)]
    if allowed_languages and "language" in candidate_df.columns:
        candidate_df = candidate_df[candidate_df["language"].isin(allowed_languages)]
    candidate_df = candidate_df.reset_index(drop=True)
    if candidate_df.empty:
        return []

    tokenized_query = normalize_query_tokens(query)
    tokenized_corpus = [normalize_query_tokens(text) for text in candidate_df["chunk_text"].tolist()]
    filtered_bm25 = BM25Okapi(tokenized_corpus)

    bm25_scores = np.array(filtered_bm25.get_scores(tokenized_query), dtype=float)
    bm25_limit = min(bm25_k, len(candidate_df))
    bm25_top_idx = np.argpartition(-bm25_scores, bm25_limit - 1)[:bm25_limit]
    bm25_top_idx = bm25_top_idx[np.argsort(-bm25_scores[bm25_top_idx])]

    q_vec = embedder.encode([query], normalize_embeddings=True)
    distances, indices = index.search(q_vec.astype(np.float32), min(vec_k * 3, len(chunks_df)))

    global_to_local = {int(row_idx): local_idx for local_idx, row_idx in enumerate(candidate_df.index)}
    vec_pairs: list[tuple[int, float]] = []
    for raw_idx, raw_score in zip(indices[0].astype(int), distances[0].astype(float)):
        if raw_idx < 0:
            continue
        if raw_idx not in global_to_local:
            continue
        vec_pairs.append((global_to_local[raw_idx], raw_score))
        if len(vec_pairs) >= vec_k:
            break

    vec_idx = np.array([idx for idx, _ in vec_pairs], dtype=int)
    vec_scores = np.array([score for _, score in vec_pairs], dtype=float)
    if getattr(index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_L2 and vec_scores.size:
        vec_scores = -vec_scores

    cand_ids = set(map(int, bm25_top_idx.tolist())) | set(map(int, vec_idx.tolist()))
    cand_ids = sorted(i for i in cand_ids if 0 <= i < len(candidate_df))
    if not cand_ids:
        return []

    bm25_norm = _minmax_norm(bm25_scores[cand_ids])
    vec_map = {int(i): float(s) for i, s in zip(vec_idx, vec_scores)}
    vec_floor = float(np.min(vec_scores)) if vec_scores.size else 0.0
    vec_norm = _minmax_norm(
        np.array([vec_map.get(int(i), vec_floor) for i in cand_ids], dtype=float)
    )

    rows: list[Chunk] = []
    for pos, idx in enumerate(cand_ids):
        row = candidate_df.iloc[idx]
        item = Chunk(
            idx=int(idx),
            doc_id=str(row["doc_id"]),
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

    rows.sort(key=lambda item: item.hybrid, reverse=True)

    deduped: list[Chunk] = []
    per_doc_counts: dict[str, int] = {}
    for row in rows:
        count = per_doc_counts.get(row.doc_id, 0)
        if count >= max_chunks_per_doc:
            continue
        per_doc_counts[row.doc_id] = count + 1
        deduped.append(row)
    return deduped


def build_context(chunks: list[Chunk], doc_lookup: dict[str, dict[str, str]], max_chunks: int) -> str:
    parts = []
    for chunk in chunks[:max_chunks]:
        title = source_title(doc_lookup, chunk.doc_id, chunk.topic)
        parts.append(f"{title} p.{chunk.page_start}-{chunk.page_end}:\n{chunk.chunk_text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    client: OpenAI,
    model: str,
    query: str,
    chunks: list[Chunk],
    doc_lookup: dict[str, dict[str, str]],
    temperature: float,
    max_chunks_for_llm: int,
    max_tokens: int,
) -> str:
    context = build_context(chunks, doc_lookup=doc_lookup, max_chunks=max_chunks_for_llm)
    prompt = f"""
You are a senior banking compliance analyst.
Audience: compliance officers, legal reviewers, and risk governance stakeholders.
Use only the context below.
Answer in the same language as the user's question.
If the context is insufficient, say that clearly and do not speculate.
Do not add outside knowledge.
Every factual claim must include a citation in this format:
(Source: TITLE pp.X-Y)

Write in a concise, professional tone.
Output sections:
1) Executive Summary
2) Compliance Implications
3) Evidence and Citations

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
    return response.choices[0].message.content or ""


st.set_page_config(page_title="Compliance Evidence Assistant", layout="wide")

docs_df = load_docs()
doc_lookup = build_doc_lookup(docs_df)
chunks_df = load_chunks()
bm25 = load_bm25(chunks_df)
embedder = load_embedder()
faiss_index = load_faiss()

available_topics = sorted(topic for topic in docs_df["topic"].dropna().unique().tolist())
available_languages = sorted(language for language in docs_df["language"].dropna().unique().tolist())

st.markdown(
    """
<style>
:root {
  --bg: #f5f5f5;
  --panel: #ffffff;
  --ink: #111111;
  --muted: #333333;
  --accent: #0b57d0;
  --border: #d0d0d0;
  --soft: #ffffff;
}
.stApp {
  background: var(--bg);
  color: var(--ink);
}
.main .block-container {
  max-width: 1180px;
  padding-top: 2rem;
  padding-bottom: 2rem;
}
.hero {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  margin-bottom: 1rem;
}
.hero-kicker {
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 0.4rem;
  font-weight: 700;
}
.hero-title {
  font-size: 2.2rem;
  line-height: 1.1;
  margin: 0;
  color: var(--ink);
  font-weight: 800;
}
.hero-copy {
  margin-top: 0.8rem;
  max-width: 760px;
  color: var(--muted);
  font-size: 1rem;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem;
}
.source-card {
  background: #fafafa;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.9rem 1rem;
  margin-bottom: 0.8rem;
}
.source-title {
  font-weight: 700;
  color: var(--ink);
}
.source-meta {
  color: var(--muted);
  font-size: 0.88rem;
  margin-top: 0.15rem;
}
.source-body {
  margin-top: 0.7rem;
  color: var(--ink);
  font-size: 0.95rem;
}
[data-testid="stSidebar"] {
  background: #f0f0f0;
  border-left: 1px solid var(--border);
}
[data-testid="stSidebar"] *,
.stMarkdown *,
label,
p,
li,
span,
div,
h1,
h2,
h3 {
  color: var(--ink) !important;
}
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stMultiSelect"] div[data-baseweb="select"],
[data-testid="stSelectbox"] div[data-baseweb="select"] {
  background: var(--soft) !important;
  color: var(--ink) !important;
  border-color: var(--border) !important;
  opacity: 1 !important;
}
[data-testid="stMultiSelect"] *,
[data-testid="stSelectbox"] *,
[data-baseweb="select"] *,
[data-baseweb="popover"] *,
[data-baseweb="menu"] * {
  color: #111111 !important;
}
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-baseweb="select"] > div,
[data-baseweb="select"] > div > div {
  background: #ffffff !important;
  color: #111111 !important;
  opacity: 1 !important;
}
[data-baseweb="select"] input {
  color: #111111 !important;
  -webkit-text-fill-color: #111111 !important;
}
[data-baseweb="select"] svg {
  fill: #111111 !important;
}
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
  color: #666666 !important;
  opacity: 1 !important;
}
[data-baseweb="tag"] {
  background: #e8f0fe !important;
  color: var(--ink) !important;
}
[data-baseweb="select"] > div {
  background: #ffffff !important;
  color: var(--ink) !important;
}
[role="listbox"] {
  background: #ffffff !important;
  border: 1px solid var(--border) !important;
  color: #111111 !important;
  opacity: 1 !important;
}
[role="option"] {
  background: #ffffff !important;
  color: #111111 !important;
  opacity: 1 !important;
}
[role="option"][aria-selected="true"] {
  background: #e8f0fe !important;
  color: #111111 !important;
}
[role="option"]:hover {
  background: #f2f6ff !important;
  color: #111111 !important;
}
ul[role="listbox"] *,
li[role="option"] * {
  color: #111111 !important;
  opacity: 1 !important;
}
button[kind="primary"] {
  background: var(--accent) !important;
  color: #ffffff !important;
  border: 1px solid var(--accent) !important;
}
button[kind="secondary"] {
  background: #ffffff !important;
  color: var(--ink) !important;
  border: 1px solid var(--border) !important;
}
[data-testid="stButton"] button {
  opacity: 1 !important;
}
[data-testid="stTabs"] button {
  color: #444444 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--ink) !important;
}
[data-testid="stDataFrame"] * {
  color: var(--ink) !important;
  opacity: 1 !important;
}
[data-testid="stAlertContainer"] * {
  color: var(--ink) !important;
}
[data-testid="stCaptionContainer"] {
  color: #555555 !important;
}
[data-baseweb="slider"] * {
  color: #111111 !important;
}
[data-baseweb="slider"] [role="slider"] {
  background: #0b57d0 !important;
  border-color: #0b57d0 !important;
  box-shadow: 0 0 0 2px #ffffff !important;
}
[data-baseweb="slider"] > div > div:first-child,
[data-baseweb="slider"] > div > div:first-child > div {
  background: #d0d0d0 !important;
}
[data-baseweb="slider"] > div > div:nth-child(2),
[data-baseweb="slider"] > div > div:nth-child(2) > div {
  background: #0b57d0 !important;
}
[data-testid="stExpander"] {
  background: #ffffff !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
[data-testid="stExpanderDetails"],
[data-testid="stExpanderDetails"] * {
  background: #ffffff !important;
  color: #111111 !important;
  opacity: 1 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-kicker">Grounded Regulatory Analysis</div>
  <h1 class="hero-title">Compliance Evidence Assistant</h1>
  <div class="hero-copy">
    Ask for a regulatory answer, and the assistant will draft a concise response using only retrieved source passages.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Assistant Settings")
    model = st.text_input("Answer model", value=DEFAULT_MODEL)
    show_sources = st.checkbox("Show supporting excerpts", value=True)
    show_scores = st.checkbox("Show retrieval scores", value=False)

    with st.expander("Advanced Retrieval", expanded=False):
        topic_filter = st.multiselect(
            "Limit to topics",
            options=available_topics,
            format_func=lambda value: TOPIC_LABELS.get(value, value.replace("_", " ").title()),
        )
        language_filter = st.multiselect("Limit to languages", options=available_languages)
        bm25_k = st.slider("Keyword candidate pool", 10, 250, 40, 10)
        vec_k = st.slider("Vector candidate pool", 10, 250, 40, 10)
        w_bm25 = st.slider("Keyword weight", 0.0, 1.0, 0.45, 0.05)
        max_chunks_for_llm = st.slider("Evidence passages sent to model", 3, 12, 6, 1)
        max_chunks_per_doc = st.slider("Max passages per source", 1, 5, 2, 1)

    with st.expander("Advanced Generation", expanded=False):
        temperature = st.slider("Temperature", 0.0, 0.5, 0.0, 0.05)
        max_tokens = st.slider("Max completion tokens", 256, 2000, 900, 50)

left_col, right_col = st.columns([1.65, 1.0], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Ask a question")
    example_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for idx, question in enumerate(EXAMPLE_QUESTIONS):
        if example_cols[idx].button(question, key=f"example_{idx}", use_container_width=True):
            st.session_state["question_input"] = question

    with st.form("ask_form", clear_on_submit=False):
        query = st.text_area(
            "Question",
            height=150,
            key="question_input",
            placeholder="Example: What are the current internal control responsibilities of the board, and how should evidence be documented?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Generate Answer", type="primary", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Enter a regulatory or compliance question to continue.")
        else:
            allowed_topics = set(topic_filter)
            allowed_languages = set(language_filter)
            retrieval_started = time.time()
            try:
                with st.spinner("Retrieving evidence..."):
                    candidates = retrieve_candidates(
                        query=query.strip(),
                        chunks_df=chunks_df,
                        bm25=bm25,
                        embedder=embedder,
                        index=faiss_index,
                        bm25_k=int(bm25_k),
                        vec_k=int(vec_k),
                        w_bm25=float(w_bm25),
                        w_vec=float(1.0 - w_bm25),
                        allowed_topics=allowed_topics,
                        allowed_languages=allowed_languages,
                        max_chunks_per_doc=int(max_chunks_per_doc),
                    )
                retrieval_latency = time.time() - retrieval_started
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")
                candidates = []
                retrieval_latency = 0.0

            if not candidates:
                st.warning("No supporting evidence was found with the current filters.")
            else:
                try:
                    with st.spinner("Drafting answer..."):
                        answer = generate_answer(
                            client=load_openai_client(),
                            model=model.strip(),
                            query=query.strip(),
                            chunks=candidates,
                            doc_lookup=doc_lookup,
                            temperature=float(temperature),
                            max_chunks_for_llm=int(max_chunks_for_llm),
                            max_tokens=int(max_tokens),
                        )
                except Exception as exc:
                    st.error(f"Answer generation failed: {exc}")
                    answer = ""

                if answer:
                    st.caption(
                        f"Prepared from {min(len(candidates), int(max_chunks_for_llm))} supporting passages in {retrieval_latency:.2f}s retrieval time."
                    )
                    st.markdown("### Compliance Response")
                    st.write(answer)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Evidence")
    st.caption("Supporting excerpts are grouped by source and pages. Raw document identifiers are intentionally hidden from the primary workflow.")

    preview_query = st.session_state.get("question_input", "").strip()
    if preview_query:
        preview_candidates = retrieve_candidates(
            query=preview_query,
            chunks_df=chunks_df,
            bm25=bm25,
            embedder=embedder,
            index=faiss_index,
            bm25_k=int(locals().get("bm25_k", 40)),
            vec_k=int(locals().get("vec_k", 40)),
            w_bm25=float(locals().get("w_bm25", 0.45)),
            w_vec=float(1.0 - float(locals().get("w_bm25", 0.45))),
            allowed_topics=set(locals().get("topic_filter", [])),
            allowed_languages=set(locals().get("language_filter", [])),
            max_chunks_per_doc=int(locals().get("max_chunks_per_doc", 2)),
        )
    else:
        preview_candidates = []

    if show_sources and preview_candidates:
        for chunk in preview_candidates[: max_chunks_for_llm if "max_chunks_for_llm" in locals() else 6]:
            title = source_title(doc_lookup, chunk.doc_id, chunk.topic)
            topic_label = TOPIC_LABELS.get(chunk.topic or "other", "Other")
            st.markdown(
                f"""
<div class="source-card">
  <div class="source-title">{title}</div>
  <div class="source-meta">{topic_label} | pages {chunk.page_start}-{chunk.page_end}</div>
  <div class="source-body">{chunk.chunk_text[:420]}{'...' if len(chunk.chunk_text) > 420 else ''}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            if show_scores:
                st.caption(
                    f"hybrid={chunk.hybrid:.4f} | keyword={chunk.bm25:.3f} | vector={chunk.vec:.3f}"
                )
    else:
        st.info("Enter a question to preview the supporting evidence that will be used.")
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Operations and Document Register", expanded=False):
    st.markdown(
        """
- Retrieval stack: BM25 + FAISS
- Embedding model: multilingual MiniLM
- Source registry: manually curated `docs.csv`
- Evidence packing: paragraph-based chunking with overlap
""".strip()
    )
    register_df = docs_df[["title", "topic", "language"]].copy()
    register_df["topic"] = register_df["topic"].map(
        lambda value: TOPIC_LABELS.get(value, str(value).replace("_", " ").title())
    )
    french_df = register_df[register_df["language"].astype(str).str.upper() == "FR"].sort_values("title")
    english_df = register_df[register_df["language"].astype(str).str.upper() == "EN"].sort_values("title")

    st.caption(
        f"Sources loaded: {len(register_df)} total | {len(french_df)} French | {len(english_df)} English"
    )
    fr_col, en_col = st.columns(2, gap="large")
    with fr_col:
        st.markdown("#### French Sources")
        if french_df.empty:
            st.write("No French sources loaded.")
        else:
            for row in french_df.itertuples(index=False):
                st.markdown(f"- {row.topic}: {row.title}")
    with en_col:
        st.markdown("#### English Sources")
        if english_df.empty:
            st.write("No English sources loaded.")
        else:
            for row in english_df.itertuples(index=False):
                st.markdown(f"- {row.topic}: {row.title}")


