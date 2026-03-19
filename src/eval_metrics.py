from __future__ import annotations

import re

REFUSAL_MESSAGE_EN = "The approved corpus does not contain enough evidence to answer with confidence."
REFUSAL_MESSAGE_FR = "Les sources approuvées ne permettent pas de répondre avec certitude."
REFUSAL_MESSAGES = (REFUSAL_MESSAGE_EN, REFUSAL_MESSAGE_FR)

CITATION_REGEX = re.compile(
    r"\(Source:\s*.+?\s+pp\.\s*\d+\-\d+\)",
    flags=re.IGNORECASE,
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "into",
    "about",
    "dans",
    "avec",
    "pour",
    "des",
    "une",
    "les",
    "sur",
    "est",
    "sont",
}


def normalize_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9À-ÖØ-öø-ÿŒœÆæ]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def keyword_recall(reference: str, candidate: str) -> float:
    ref_tokens = normalize_tokens(reference)
    if not ref_tokens:
        return 0.0
    cand_tokens = normalize_tokens(candidate)
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


def has_citation(text: str) -> int:
    return int(bool(CITATION_REGEX.search(text or "")))


def is_refusal(text: str) -> int:
    lowered = (text or "").lower()
    return int(any(message.lower() in lowered for message in REFUSAL_MESSAGES))
