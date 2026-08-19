from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from openai import OpenAI

CITATION_PATTERN = re.compile(
    r"\(Source:\s*(?P<doc_id>.+?)\s+pp?\.\s*(?P<page_start>\d+)"
    r"(?:\s*-\s*(?P<page_end>\d+))?\)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Citation:
    doc_id: str
    page_start: int
    page_end: int


@dataclass
class ClaimCheck:
    claim_index: int
    text: str
    citations: list[Citation] = field(default_factory=list)
    provenance_valid: bool = False
    supported: bool | None = None
    reason: str = ""


@dataclass
class VerificationReport:
    valid: bool
    is_refusal: bool
    semantic_checked: bool
    claims: list[ClaimCheck]
    errors: list[str]
    answer_relevance: bool | None = None

    @property
    def verified_claims(self) -> list[ClaimCheck]:
        """Claims that are safe to return independently."""
        return [
            claim
            for claim in self.claims
            if claim.citations and claim.provenance_valid and claim.supported is True
        ]

    @property
    def rejected_claims(self) -> list[ClaimCheck]:
        """Claims that failed citation provenance or semantic support."""
        return [
            claim
            for claim in self.claims
            if not (claim.citations and claim.provenance_valid and claim.supported is True)
        ]

    @property
    def can_return_partial(self) -> bool:
        """Whether a filtered answer remains relevant and contains verified claims."""
        return (
            not self.valid
            and self.semantic_checked
            and self.answer_relevance is True
            and bool(self.verified_claims)
        )

    @property
    def citation_coverage(self) -> float | None:
        if not self.claims:
            return None if self.is_refusal else 0.0
        return sum(bool(claim.citations) for claim in self.claims) / len(self.claims)

    @property
    def provenance_accuracy(self) -> float | None:
        if not self.claims:
            return None if self.is_refusal else 0.0
        return sum(claim.provenance_valid for claim in self.claims) / len(self.claims)

    @property
    def support_rate(self) -> float | None:
        checked = [claim for claim in self.claims if claim.supported is not None]
        if not checked:
            return None
        return sum(bool(claim.supported) for claim in checked) / len(checked)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citation_coverage"] = self.citation_coverage
        payload["provenance_accuracy"] = self.provenance_accuracy
        payload["support_rate"] = self.support_rate
        payload["verified_claim_count"] = len(self.verified_claims)
        payload["rejected_claim_count"] = len(self.rejected_claims)
        payload["can_return_partial"] = self.can_return_partial
        return payload


def _is_refusal(text: str) -> bool:
    normalized = text.lower()
    markers = (
        "sources fournies ne permettent pas",
        "provided sources do not support",
        "insufficient evidence",
        "preuves insuffisantes",
    )
    return any(marker in normalized for marker in markers)


def extract_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for raw_line in answer.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line).strip()
        if not line or line.startswith("#") or re.fullmatch(r"[A-ZÀ-ÖØ-Ý _-]+:?", line):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])", line):
            sentence = sentence.strip()
            words = re.findall(r"\w+", CITATION_PATTERN.sub("", sentence))
            if len(words) >= 4:
                claims.append(sentence)
    return claims


def parse_citations(text: str) -> list[Citation]:
    citations: list[Citation] = []
    for match in CITATION_PATTERN.finditer(text):
        page_start = int(match.group("page_start"))
        page_end = int(match.group("page_end") or page_start)
        citations.append(
            Citation(
                doc_id=match.group("doc_id").strip(),
                page_start=page_start,
                page_end=page_end,
            )
        )
    return citations


def _evidence_rows(evidence: Any) -> list[dict[str, Any]]:
    if hasattr(evidence, "to_dict"):
        return list(evidence.to_dict("records"))
    rows: list[dict[str, Any]] = []
    for item in evidence:
        if isinstance(item, Mapping):
            rows.append(dict(item))
        else:
            rows.append(
                {
                    "doc_id": item.doc_id,
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "chunk_text": item.chunk_text,
                }
            )
    return rows


def _matching_evidence(citation: Citation, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row["doc_id"]) == citation.doc_id
        and citation.page_start >= int(row["page_start"])
        and citation.page_end <= int(row["page_end"])
    ]


def _semantic_checks(
    client: OpenAI,
    model: str,
    claims: list[ClaimCheck],
    evidence_rows: list[dict[str, Any]],
    question: str | None,
) -> tuple[dict[int, tuple[bool, str]], bool, str]:
    items = []
    evidence_catalog: list[dict[str, Any]] = []
    evidence_ids: dict[tuple[str, int, int, str], str] = {}
    for claim in claims:
        claim_evidence_ids: list[str] = []
        for citation in claim.citations:
            for row in _matching_evidence(citation, evidence_rows):
                key = (
                    str(row["doc_id"]),
                    int(row["page_start"]),
                    int(row["page_end"]),
                    str(row["chunk_text"]),
                )
                evidence_id = evidence_ids.get(key)
                if evidence_id is None:
                    evidence_id = f"E{len(evidence_catalog) + 1}"
                    evidence_ids[key] = evidence_id
                    evidence_catalog.append(
                        {
                            "evidence_id": evidence_id,
                            "source": key[0],
                            "pages": f"{key[1]}-{key[2]}",
                            "text": key[3],
                        }
                    )
                if evidence_id not in claim_evidence_ids:
                    claim_evidence_ids.append(evidence_id)
        items.append(
            {
                "claim_index": claim.claim_index,
                "claim": claim.text,
                "citations": [asdict(citation) for citation in claim.citations],
                "evidence_ids": claim_evidence_ids,
            }
        )

    schema = {
        "type": "object",
        "properties": {
            "answers_question": {"type": "boolean"},
            "answer_reason": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_index": {"type": "integer"},
                        "supported": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["claim_index", "supported", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["answers_question", "answer_reason", "claims"],
        "additionalProperties": False,
    }
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_completion_tokens=700,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "citation_verification", "strict": True, "schema": schema},
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You verify whether each claim is directly supported by its cited evidence. "
                    "Use only the supplied excerpts. A faithful paraphrase or direct conclusion is supported; "
                    "a merely related statement, contradiction, or material addition is not. Judge the factual "
                    "content of each claim, not whether it repeats context already established by the question "
                    "or cited legal instrument. In particular, do not reject an otherwise supported claim only "
                    "because it does not repeat a jurisdiction, entity, date, or product that is unambiguous from "
                    "that context. Then judge answers_question using the supported claims together. Set it to "
                    "true only when those claims provide a useful answer to the question as a whole. Set it to "
                    "false when the answer changes, conflicts with, or genuinely ignores a material restriction."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "claims": items,
                        "evidence": evidence_catalog,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    checks = {
        int(item["claim_index"]): (bool(item["supported"]), str(item["reason"]))
        for item in payload.get("claims", [])
    }
    return checks, bool(payload.get("answers_question")), str(payload.get("answer_reason", ""))


def verify_citations(
    answer: str,
    evidence: Any,
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    semantic: bool = False,
    question: str | None = None,
) -> VerificationReport:
    refusal = _is_refusal(answer)
    claim_texts = [] if refusal else extract_claims(answer)
    rows = _evidence_rows(evidence)
    checks: list[ClaimCheck] = []
    errors: list[str] = []

    for index, text in enumerate(claim_texts):
        citations = parse_citations(text)
        provenance_valid = bool(citations) and all(
            _matching_evidence(citation, rows) for citation in citations
        )
        if not citations:
            errors.append(f"Claim {index} has no citation.")
        elif not provenance_valid:
            errors.append(f"Claim {index} cites evidence not present in the supplied context.")
        checks.append(
            ClaimCheck(
                claim_index=index,
                text=text,
                citations=citations,
                provenance_valid=provenance_valid,
            )
        )

    semantic_checked = False
    answer_relevance: bool | None = None
    if semantic and checks:
        if client is None or not model:
            errors.append("Semantic verification was requested without a client and model.")
        else:
            try:
                semantic_results, answer_relevance, answer_reason = _semantic_checks(
                    client, model, checks, rows, question
                )
                semantic_checked = True
                for check in checks:
                    check.supported, check.reason = semantic_results.get(
                        check.claim_index, (False, "Verifier omitted this claim.")
                    )
                    if not check.supported:
                        errors.append(f"Claim {check.claim_index} is not supported: {check.reason}")
                if question and not answer_relevance:
                    errors.append(f"The answer does not address the question: {answer_reason}")
            except Exception as exc:
                errors.append(f"Semantic verification failed: {type(exc).__name__}")

    if not refusal and not checks:
        errors.append("The answer contains no verifiable claims.")
    if semantic and checks and not semantic_checked and not any(
        error.startswith("Semantic verification failed") for error in errors
    ):
        errors.append("Semantic verification did not run.")

    return VerificationReport(
        valid=not errors,
        is_refusal=refusal,
        semantic_checked=semantic_checked,
        claims=checks,
        errors=errors,
        answer_relevance=answer_relevance,
    )
