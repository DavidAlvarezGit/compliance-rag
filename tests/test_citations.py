import json
from types import SimpleNamespace

import pandas as pd

from src.answer import select_verified_answer
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
    assert not report.can_return_partial
    assert select_verified_answer(answer, report, "What rules apply on Mars?").startswith(
        "The provided sources do not support"
    )
    assert any("does not address the question" in error for error in report.errors)


def test_unsupported_claim_is_removed_while_verified_claim_survives() -> None:
    payload = {
        "answers_question": True,
        "answer_reason": "The supported claim answers the question.",
        "claims": [
            {"claim_index": 0, "supported": True, "reason": "Directly stated."},
            {"claim_index": 1, "supported": False, "reason": "Not stated."},
        ],
    }
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )
    supported = (
        "The board oversees risk governance and internal controls. "
        "(Source: governance-doc pp.10-12)"
    )
    unsupported = "The board must meet every week. (Source: governance-doc pp.10-12)"
    draft = f"- {supported}\n- {unsupported}"

    report = verify_citations(
        draft,
        evidence(),
        client=client,
        model="fake-verifier",
        semantic=True,
        question="What does the board oversee?",
    )
    answer = select_verified_answer(draft, report, "What does the board oversee?")

    assert not report.valid
    assert report.can_return_partial
    assert len(report.verified_claims) == 1
    assert len(report.rejected_claims) == 1
    assert supported in answer
    assert unsupported not in answer


def test_invalid_citation_does_not_block_verification_of_other_claims() -> None:
    payload = {
        "answers_question": True,
        "answer_reason": "The valid claim answers the question.",
        "claims": [
            {"claim_index": 0, "supported": True, "reason": "Directly stated."},
            {"claim_index": 1, "supported": False, "reason": "No supplied evidence."},
        ],
    }
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )
    supported = (
        "The board oversees risk governance and internal controls. "
        "(Source: governance-doc pp.10-12)"
    )
    draft = f"- {supported}\n- The board must meet every week."

    report = verify_citations(
        draft,
        evidence(),
        client=client,
        model="fake-verifier",
        semantic=True,
        question="What does the board oversee?",
    )

    assert report.semantic_checked
    assert report.can_return_partial
    assert select_verified_answer(draft, report, "What does the board oversee?") == f"- {supported}"


def test_semantic_verifier_deduplicates_shared_evidence() -> None:
    captured: dict = {}
    payload = {
        "answers_question": True,
        "answer_reason": "Both claims answer the question.",
        "claims": [
            {"claim_index": 0, "supported": True, "reason": "Directly stated."},
            {"claim_index": 1, "supported": True, "reason": "Directly stated."},
        ],
    }
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )

    def create(**kwargs):
        captured.update(kwargs)
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    citation = "(Source: governance-doc pp.10-12)"
    answer = (
        f"- The board oversees risk governance. {citation}\n"
        f"- The board oversees internal controls. {citation}"
    )

    report = verify_citations(
        answer,
        evidence(),
        client=client,
        model="fake-verifier",
        semantic=True,
        question="What does the board oversee?",
    )
    request = json.loads(captured["messages"][1]["content"])
    verifier_instructions = captured["messages"][0]["content"].lower()

    assert report.valid
    assert len(request["evidence"]) == 1
    assert request["claims"][0]["evidence_ids"] == ["E1"]
    assert request["claims"][1]["evidence_ids"] == ["E1"]
    assert "not whether it repeats context" in verifier_instructions
    assert "do not reject an otherwise supported claim" in verifier_instructions
