import json
from types import SimpleNamespace

import pandas as pd

from src.citations import extract_claims, parse_citations, verify_citations


def evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "doc_id": "governance-doc",
                "page_start": 10,
                "page_end": 12,
                "chunk_text": "The board oversees risk governance and internal controls.",
            }
        ]
    )


def test_claim_citations_are_checked_against_supplied_pages() -> None:
    answer = (
        "- The board oversees risk governance and internal controls. "
        "(Source: governance-doc pp.10-12)"
    )

    report = verify_citations(answer, evidence())

    assert report.valid
    assert report.citation_coverage == 1.0
    assert report.provenance_accuracy == 1.0
    assert parse_citations(answer)[0].doc_id == "governance-doc"


def test_missing_or_out_of_context_citations_fail_closed() -> None:
    missing = verify_citations("The board oversees all risks.", evidence())
    wrong_page = verify_citations(
        "The board oversees all risks. (Source: governance-doc pp.99-100)", evidence()
    )

    assert not missing.valid
    assert not wrong_page.valid
    assert extract_claims("The board oversees all risks.")


def test_refusal_requires_no_citation() -> None:
    report = verify_citations(
        "The provided sources do not support a sufficiently certain answer.", evidence()
    )

    assert report.valid
    assert report.is_refusal
    assert report.citation_coverage is None
    assert report.provenance_accuracy is None


def test_material_question_constraint_failure_invalidates_supported_claims() -> None:
    payload = {
        "answers_question": False,
        "answer_reason": "The evidence does not cover the requested location.",
        "claims": [{"claim_index": 0, "supported": True, "reason": "The generic claim is supported."}],
    }
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )
    completions = SimpleNamespace(create=lambda **_: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    answer = (
        "Mortgage collateral must be valued prudently. "
        "(Source: governance-doc pp.10-12)"
    )

    report = verify_citations(
        answer,
        evidence(),
        client=client,
        model="fake-verifier",
        semantic=True,
        question="What mortgage collateral rules apply on Mars?",
    )

    assert not report.valid
    assert report.answer_relevance is False
    assert any("does not address the question" in error for error in report.errors)
