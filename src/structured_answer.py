from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


DOC_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class AnswerCitation:
    doc_id: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class AnswerClaim:
    text: str
    citations: tuple[AnswerCitation, ...]


@dataclass(frozen=True)
class StructuredAnswer:
    refusal: bool
    claims: tuple[AnswerClaim, ...]


def structured_answer_response_format() -> dict[str, Any]:
    """Return the strict JSON schema used for grounded answer generation."""
    citation_schema = {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string"},
            "page_start": {"type": "integer"},
            "page_end": {"type": "integer"},
        },
        "required": ["doc_id", "page_start", "page_end"],
        "additionalProperties": False,
    }
    claim_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "citations": {"type": "array", "items": citation_schema},
        },
        "required": ["text", "citations"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "grounded_regulatory_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "refusal": {"type": "boolean"},
                    "claims": {"type": "array", "items": claim_schema},
                },
                "required": ["refusal", "claims"],
                "additionalProperties": False,
            },
        },
    }


def parse_structured_answer(raw: str) -> StructuredAnswer:
    """Validate model JSON before it can be rendered or verified."""
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("The answer model returned invalid JSON.") from exc

    if not isinstance(payload, dict) or set(payload) != {"refusal", "claims"}:
        raise ValueError("The structured answer has an invalid top-level shape.")
    if not isinstance(payload["refusal"], bool) or not isinstance(payload["claims"], list):
        raise ValueError("The structured answer contains invalid field types.")

    claims: list[AnswerClaim] = []
    for claim_payload in payload["claims"]:
        if not isinstance(claim_payload, dict) or set(claim_payload) != {"text", "citations"}:
            raise ValueError("A structured claim has an invalid shape.")
        text = claim_payload["text"]
        citation_payloads = claim_payload["citations"]
        if not isinstance(text, str) or len(re.findall(r"\w+", text)) < 4:
            raise ValueError("A structured claim has insufficient text.")
        if "(Source:" in text:
            raise ValueError("Claim text must not contain rendered citations.")
        if not isinstance(citation_payloads, list) or not citation_payloads:
            raise ValueError("Every substantive claim requires at least one citation.")

        citations: list[AnswerCitation] = []
        for citation_payload in citation_payloads:
            if not isinstance(citation_payload, dict) or set(citation_payload) != {
                "doc_id",
                "page_start",
                "page_end",
            }:
                raise ValueError("A structured citation has an invalid shape.")
            doc_id = citation_payload["doc_id"]
            page_start = citation_payload["page_start"]
            page_end = citation_payload["page_end"]
            if not isinstance(doc_id, str) or not DOC_ID_PATTERN.fullmatch(doc_id):
                raise ValueError("A structured citation contains an invalid document ID.")
            if (
                isinstance(page_start, bool)
                or isinstance(page_end, bool)
                or not isinstance(page_start, int)
                or not isinstance(page_end, int)
                or page_start < 1
                or page_end < page_start
            ):
                raise ValueError("A structured citation contains an invalid page range.")
            citations.append(AnswerCitation(doc_id, page_start, page_end))
        claims.append(AnswerClaim(text.strip(), tuple(citations)))

    if payload["refusal"]:
        if claims:
            raise ValueError("A refusal cannot contain substantive claims.")
    elif not claims:
        raise ValueError("A substantive answer must contain at least one claim.")
    elif len(claims) > 4:
        raise ValueError("A structured answer cannot contain more than four claims.")

    return StructuredAnswer(refusal=payload["refusal"], claims=tuple(claims))


def render_structured_answer(answer: StructuredAnswer) -> str:
    """Render validated claims with deterministic citations."""
    return "\n\n".join(render_structured_claims(answer))


def render_structured_claims(answer: StructuredAnswer) -> tuple[str, ...]:
    """Render each validated claim as one independent verification unit."""
    if answer.refusal:
        raise ValueError("Refusals must use the application's localized refusal message.")

    paragraphs: list[str] = []
    for claim in answer.claims:
        citations = " ".join(
            f"(Source: {citation.doc_id} pp.{citation.page_start}-{citation.page_end})"
            for citation in claim.citations
        )
        paragraphs.append(f"{claim.text} {citations}")
    return tuple(paragraphs)


def render_model_output(raw: str, refusal_text: str) -> tuple[str, tuple[str, ...]]:
    """Fail closed and render a model response for verification and display."""
    try:
        answer = parse_structured_answer(raw)
    except ValueError:
        return refusal_text, ()
    if answer.refusal:
        return refusal_text, ()
    rendered_claims = render_structured_claims(answer)
    return "\n\n".join(rendered_claims), rendered_claims
