from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .artifacts import (
    ARTIFACT_MANIFEST_PATH,
    CHUNKS_PATH as DEFAULT_CHUNKS_PATH,
    FAISS_INDEX_PATH as DEFAULT_FAISS_INDEX_PATH,
    validate_artifact_manifest,
)
from .eval_metrics import REFUSAL_MESSAGE_EN, REFUSAL_MESSAGE_FR

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "data" / "metadata" / "docs.csv"
load_dotenv(BASE_DIR / ".env")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

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

FRENCH_HINTS = {
    "quelles",
    "quelle",
    "risque",
    "liquidite",
    "liquidité",
    "banque",
    "sources",
    "obligations",
    "gouvernance",
    "selon",
    "corpus",
}


@dataclass(frozen=True)
class RetrievalConfig:
    bm25_k: int = 40
    vec_k: int = 40
    w_bm25: float = 0.45
    allowed_topics: frozenset[str] = frozenset()
    allowed_languages: frozenset[str] = frozenset()
    max_chunks_per_doc: int = 2
    top_k: int = 8


@dataclass(frozen=True)
class GenerationConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_chunks_for_llm: int = 8
    max_tokens: int = 1500


@dataclass(frozen=True)
class ChunkResult:
    row_id: int
    doc_id: str
    page_start: int
    page_end: int
    topic: Optional[str]
    issue: Optional[str]
    language: Optional[str]
    chunk_text: str
    bm25: float
    vec: float
    hybrid: float


@dataclass(frozen=True)
class RAGResources:
    docs_df: pd.DataFrame
    doc_lookup: dict[str, dict[str, str]]
    chunks_df: pd.DataFrame
    embedder: SentenceTransformer
    index: faiss.Index


def normalize_query_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    return re.findall(r"[a-z0-9]+", normalized)


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _minmax_norm(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    xmin = float(np.min(values))
    xmax = float(np.max(values))
    if abs(xmax - xmin) < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - xmin) / (xmax - xmin)


def load_docs(path: Path = DOCS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8").copy()
    expected = {"doc_id", "title", "topic", "language"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"docs.csv missing columns: {sorted(missing)}")
    return df


def load_chunks(path: Path = DEFAULT_CHUNKS_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    required = {"chunk_text", "doc_id", "page_start", "page_end"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"chunks.parquet missing columns: {sorted(missing)}")
    df["page_start"] = df["page_start"].apply(lambda value: safe_int(value, 0))
    df["page_end"] = df["page_end"].apply(lambda value: safe_int(value, 0))
    if "topic" not in df.columns:
        df["topic"] = None
    if "issue" not in df.columns:
        df["issue"] = None
    if "language" not in df.columns:
        df["language"] = None
    return df.reset_index(drop=True)


def build_doc_lookup(docs_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in docs_df.itertuples(index=False):
        lookup[str(row.doc_id)] = {
            "title": str(row.title),
            "topic": str(row.topic),
            "language": str(row.language),
        }
    return lookup


@lru_cache(maxsize=1)
def load_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the embedding model. Ensure it is cached locally or allow network access "
            f"for the first download: {model_name}"
        ) from exc


@lru_cache(maxsize=1)
def load_faiss_index(index_path: str = str(DEFAULT_FAISS_INDEX_PATH)) -> faiss.Index:
    return faiss.read_index(index_path)


@lru_cache(maxsize=1)
def load_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def load_resources(
    docs_path: str = str(DOCS_PATH),
    chunks_path: str = str(DEFAULT_CHUNKS_PATH),
    index_path: str = str(DEFAULT_FAISS_INDEX_PATH),
    embedding_model: str = EMBEDDING_MODEL,
    manifest_path: str = str(ARTIFACT_MANIFEST_PATH),
) -> RAGResources:
    validate_artifact_manifest(
        chunks_path=Path(chunks_path),
        faiss_index_path=Path(index_path),
        manifest_path=Path(manifest_path),
        embedding_model=embedding_model,
    )
    docs_df = load_docs(Path(docs_path))
    return RAGResources(
        docs_df=docs_df,
        doc_lookup=build_doc_lookup(docs_df),
        chunks_df=load_chunks(Path(chunks_path)),
        embedder=load_embedder(embedding_model),
        index=load_faiss_index(index_path),
    )


def source_title(doc_lookup: dict[str, dict[str, str]], doc_id: str, topic: Optional[str]) -> str:
    if doc_id in doc_lookup:
        title = doc_lookup[doc_id].get("title", "").strip()
        if title:
            return title
    if topic:
        return TOPIC_LABELS.get(topic, topic.replace("_", " ").title())
    return doc_id


def detect_query_language(query: str) -> str:
    tokens = set(normalize_query_tokens(query))
    return "fr" if tokens & FRENCH_HINTS else "en"


def refusal_message_for_query(query: str) -> str:
    return REFUSAL_MESSAGE_FR if detect_query_language(query) == "fr" else REFUSAL_MESSAGE_EN


def retrieve_candidates(
    query: str,
    *,
    chunks_df: pd.DataFrame,
    embedder: SentenceTransformer,
    index: faiss.Index,
    config: RetrievalConfig,
) -> list[ChunkResult]:
    candidate_df = chunks_df
    if config.allowed_topics:
        candidate_df = candidate_df[candidate_df["topic"].isin(config.allowed_topics)]
    if config.allowed_languages:
        candidate_df = candidate_df[candidate_df["language"].isin(config.allowed_languages)]
    if candidate_df.empty:
        return []

    candidate_global_ids = candidate_df.index.to_numpy(dtype=int)
    tokenized_corpus = [
        normalize_query_tokens(text)
        for text in candidate_df["chunk_text"].astype(str).tolist()
    ]
    filtered_bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = normalize_query_tokens(query)
    bm25_scores = np.array(filtered_bm25.get_scores(tokenized_query), dtype=float)

    bm25_limit = min(config.bm25_k, len(candidate_df))
    bm25_top_pos = np.argpartition(-bm25_scores, bm25_limit - 1)[:bm25_limit]
    bm25_top_pos = bm25_top_pos[np.argsort(-bm25_scores[bm25_top_pos])]
    bm25_raw_map = {
        int(candidate_global_ids[pos]): float(bm25_scores[pos])
        for pos in bm25_top_pos.tolist()
    }

    q_vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    vec_limit = min(max(config.vec_k * 3, config.vec_k), len(chunks_df))
    distances, indices = index.search(q_vec, vec_limit)
    candidate_id_set = set(map(int, candidate_global_ids.tolist()))
    vec_raw_map: dict[int, float] = {}
    for raw_idx, raw_score in zip(indices[0].astype(int), distances[0].astype(float)):
        if raw_idx < 0 or raw_idx not in candidate_id_set:
            continue
        vec_raw_map[int(raw_idx)] = float(raw_score)
        if len(vec_raw_map) >= config.vec_k:
            break

    vec_scores = np.array(list(vec_raw_map.values()), dtype=float)
    if getattr(index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_L2 and vec_scores.size:
        vec_scores = -vec_scores
        vec_raw_map = {
            idx: float(score)
            for idx, score in zip(vec_raw_map.keys(), vec_scores.tolist())
        }

    candidate_ids = sorted(set(bm25_raw_map) | set(vec_raw_map))
    if not candidate_ids:
        return []

    bm25_values = np.array([bm25_raw_map.get(idx, 0.0) for idx in candidate_ids], dtype=float)
    vec_floor = float(np.min(vec_scores)) if vec_scores.size else 0.0
    vec_values = np.array([vec_raw_map.get(idx, vec_floor) for idx in candidate_ids], dtype=float)
    bm25_norm = _minmax_norm(bm25_values)
    vec_norm = _minmax_norm(vec_values)

    rows: list[ChunkResult] = []
    for pos, global_idx in enumerate(candidate_ids):
        row = chunks_df.iloc[int(global_idx)]
        rows.append(
            ChunkResult(
                row_id=int(global_idx),
                doc_id=str(row["doc_id"]),
                page_start=safe_int(row["page_start"]),
                page_end=safe_int(row["page_end"]),
                topic=None if pd.isna(row.get("topic", None)) else str(row.get("topic", None)),
                issue=None if pd.isna(row.get("issue", None)) else str(row.get("issue", None)),
                language=None if pd.isna(row.get("language", None)) else str(row.get("language", None)),
                chunk_text=str(row["chunk_text"]),
                bm25=float(bm25_norm[pos]),
                vec=float(vec_norm[pos]),
                hybrid=float(config.w_bm25 * bm25_norm[pos] + (1.0 - config.w_bm25) * vec_norm[pos]),
            )
        )

    rows.sort(key=lambda item: item.hybrid, reverse=True)
    deduped: list[ChunkResult] = []
    per_doc_counts: dict[str, int] = {}
    for row in rows:
        count = per_doc_counts.get(row.doc_id, 0)
        if count >= config.max_chunks_per_doc:
            continue
        per_doc_counts[row.doc_id] = count + 1
        deduped.append(row)
        if len(deduped) >= config.top_k:
            break
    return deduped


def build_context(
    chunks: list[ChunkResult],
    *,
    doc_lookup: dict[str, dict[str, str]],
    max_chunks: int,
) -> str:
    blocks = []
    for chunk in chunks[:max_chunks]:
        title = source_title(doc_lookup, chunk.doc_id, chunk.topic)
        blocks.append(f"{title} pp.{chunk.page_start}-{chunk.page_end}:\n{chunk.chunk_text}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(
    query: str,
    chunks: list[ChunkResult],
    *,
    doc_lookup: dict[str, dict[str, str]],
    client: Optional[OpenAI] = None,
    config: Optional[GenerationConfig] = None,
) -> str:
    config = config or GenerationConfig()
    refusal_message = refusal_message_for_query(query)
    prompt = f"""
You are a senior banking compliance analyst.
Audience: compliance officers, legal reviewers, and risk governance stakeholders.
Use only the context below.
Answer in the same language as the user's question.
If the context is insufficient, answer with exactly:
{refusal_message}
Do not add outside knowledge.
Do not speculate.
Every factual claim must include a citation in this format:
(Source: TITLE pp.X-Y)

Write in a concise but sufficiently informative professional tone.
Output sections:
1) Executive Summary
2) Compliance Implications
3) Evidence and Citations

CONTEXT:
{build_context(chunks, doc_lookup=doc_lookup, max_chunks=config.max_chunks_for_llm)}

QUESTION:
{query}
""".strip()
    client = client or load_openai_client()
    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous banking regulation analyst. "
                    "You must stay grounded in the provided evidence."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def answer_question(
    query: str,
    *,
    retrieval_config: Optional[RetrievalConfig] = None,
    generation_config: Optional[GenerationConfig] = None,
) -> str:
    resources = load_resources()
    retrieval_config = retrieval_config or RetrievalConfig()
    chunks = retrieve_candidates(
        query,
        chunks_df=resources.chunks_df,
        embedder=resources.embedder,
        index=resources.index,
        config=retrieval_config,
    )
    if not chunks:
        return refusal_message_for_query(query)
    return generate_answer(
        query,
        chunks,
        doc_lookup=resources.doc_lookup,
        config=generation_config,
    )
