from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import (
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    GenerationConfig,
    RetrievalConfig,
    TOPIC_LABELS,
    build_doc_lookup,
    load_docs,
    load_openai_client,
    load_resources,
    retrieve_candidates,
    generate_answer,
)

load_dotenv(PROJECT_ROOT / ".env")

EXAMPLE_QUESTIONS = [
    "What governance responsibilities does the board have for internal controls?",
    "How should trading-related operational risk incidents be escalated and managed?",
    "Quelles sont les obligations principales en matière de risque de liquidité ?",
    "What does the current corpus say about climate and nature-related financial risk governance?",
]


def render_register_table(df):
    rows = []
    for row in df.itertuples(index=False):
        rows.append(f"<tr><td>{row.title}</td><td>{row.topic}</td></tr>")
    return """
<table class="register-table">
  <thead>
    <tr><th>Source</th><th>Topic</th></tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
""".strip().format(rows="".join(rows))


st.set_page_config(page_title="Compliance Evidence Assistant", layout="wide")
docs_path = os.getenv("DOCS_PATH", str(PROJECT_ROOT / "data" / "metadata" / "docs.csv"))
chunks_path = os.getenv("CHUNKS_PATH", str(PROJECT_ROOT / "data" / "processed" / "chunks.parquet"))
index_path = os.getenv("FAISS_INDEX_PATH", str(PROJECT_ROOT / "data" / "artifacts" / "faiss.index"))
embedding_model = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
docs_df = load_docs(Path(docs_path))
doc_lookup = build_doc_lookup(docs_df)
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
.register-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 0.35rem;
}
.register-table th,
.register-table td {
  text-align: left;
  vertical-align: top;
  padding: 0.55rem 0.4rem;
  border-bottom: 1px solid var(--border);
  color: #111111;
}
.register-table th {
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #444444;
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
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
  color: #666666 !important;
  opacity: 1 !important;
}
[data-baseweb="tag"] {
  background: #e8f0fe !important;
  color: var(--ink) !important;
}
button[kind="primary"] {
  background: var(--accent) !important;
  color: #ffffff !important;
  border: 1px solid var(--accent) !important;
}
[data-testid="stExpander"] {
  background: #ffffff !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
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
        top_k = st.slider("Retrieved passages", 3, 12, 8, 1)
        max_chunks_per_doc = st.slider("Max passages per source", 1, 5, 2, 1)

    with st.expander("Advanced Generation", expanded=False):
        temperature = st.slider("Temperature", 0.0, 0.5, 0.0, 0.05)
        max_tokens = st.slider("Max completion tokens", 256, 2000, 1500, 50)

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
        retrieval_started = time.time()
        try:
            with st.spinner("Retrieving evidence..."):
                resources = load_resources(
                    docs_path=docs_path,
                    chunks_path=chunks_path,
                    index_path=index_path,
                    embedding_model=embedding_model,
                )
                candidates = retrieve_candidates(
                    query=query.strip(),
                    chunks_df=resources.chunks_df,
                    embedder=resources.embedder,
                    index=resources.index,
                    config=RetrievalConfig(
                        bm25_k=int(bm25_k),
                        vec_k=int(vec_k),
                        w_bm25=float(w_bm25),
                        allowed_topics=frozenset(topic_filter),
                        allowed_languages=frozenset(language_filter),
                        max_chunks_per_doc=int(max_chunks_per_doc),
                        top_k=int(top_k),
                    ),
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
                        query=query.strip(),
                        chunks=candidates,
                        doc_lookup=doc_lookup,
                        client=load_openai_client(),
                        config=GenerationConfig(
                            model=model.strip(),
                            temperature=float(temperature),
                            max_chunks_for_llm=int(top_k),
                            max_tokens=int(max_tokens),
                        ),
                    )
            except Exception as exc:
                st.error(f"Answer generation failed: {exc}")
                answer = ""

            if answer:
                st.caption(
                    f"Prepared from {len(candidates)} supporting passages in {retrieval_latency:.2f}s retrieval time."
                )
                st.markdown("### Compliance Response")
                st.write(answer)

                if show_sources:
                    st.markdown("### Supporting Excerpts")
                    for chunk in candidates:
                        title = doc_lookup.get(chunk.doc_id, {}).get("title", chunk.doc_id)
                        st.markdown(
                            f"""
<div class="source-card">
  <div class="source-title">{title}</div>
  <div class="source-meta">Pages {chunk.page_start}-{chunk.page_end}</div>
  <div class="source-body">{chunk.chunk_text}</div>
</div>
""",
                            unsafe_allow_html=True,
                        )

                if show_scores:
                    st.markdown("### Retrieval Scores")
                    st.dataframe(
                        [
                            {
                                "doc_id": chunk.doc_id,
                                "pages": f"{chunk.page_start}-{chunk.page_end}",
                                "bm25": round(chunk.bm25, 3),
                                "vector": round(chunk.vec, 3),
                                "hybrid": round(chunk.hybrid, 3),
                            }
                            for chunk in candidates
                        ],
                        use_container_width=True,
                    )
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
            st.markdown(render_register_table(french_df), unsafe_allow_html=True)
    with en_col:
        st.markdown("#### English Sources")
        if english_df.empty:
            st.write("No English sources loaded.")
        else:
            st.markdown(render_register_table(english_df), unsafe_allow_html=True)
