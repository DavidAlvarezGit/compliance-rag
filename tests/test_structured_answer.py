import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src import answer as answer_module
from src.citations import verify_citations
from src.structured_answer import (
    parse_structured_answer,
    render_model_output,
    render_structured_answer,
    render_structured_claims,
    structured_answer_response_format,
)


def payload(*, text: str = "The board oversees internal controls.") -> str:
    return json.dumps(
        {
            "refusal": False,
            "claims": [
                {
                    "text": text,
                    "citations": [
                        {
                            "doc_id": "governance-doc",
                            "page_start": 10,
                            "page_end": 12,
                        }
                    ],
                }
            ],
        }
    )


def evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "doc_id": "governance-doc",
                "page_start": 10,
                "page_end": 12,
                "chunk_text": "The board oversees internal controls and their effectiveness.",
            }
        ]
    )


def test_english_claim_is_rendered_with_a_deterministic_citation() -> None:
    answer = parse_structured_answer(payload())

    rendered = render_structured_answer(answer)

    assert rendered == (
        "The board oversees internal controls. "
        "(Source: governance-doc pp.10-12)"
    )


def test_french_claim_text_is_preserved() -> None:
    answer = parse_structured_answer(
        payload(text="Les banques doivent disposer de liquidités suffisantes.")
    )

    assert render_structured_answer(answer).startswith(
        "Les banques doivent disposer de liquidités suffisantes."
    )


def test_refusal_has_no_claims_and_must_use_localized_application_copy() -> None:
    answer = parse_structured_answer(json.dumps({"refusal": True, "claims": []}))

    assert answer.refusal
    with pytest.raises(ValueError, match="localized refusal"):
        render_structured_answer(answer)


def test_model_refusal_uses_the_supplied_localized_message() -> None:
    rendered, claims = render_model_output(
        json.dumps({"refusal": True, "claims": []}),
        "Les sources sont insuffisantes.",
    )

    assert rendered == "Les sources sont insuffisantes."
    assert claims == ()


def test_invalid_model_json_fails_closed() -> None:
    rendered, claims = render_model_output(
        "not-json", "The sources are insufficient."
    )

    assert rendered == "The sources are insufficient."
    assert claims == ()


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        json.dumps({"refusal": False}),
        json.dumps({"refusal": False, "claims": []}),
        json.dumps(
            {
                "refusal": False,
                "claims": [{"text": "Too short", "citations": []}],
            }
        ),
        json.dumps(
            {
                "refusal": False,
                "claims": [
                    {
                        "text": "A sufficiently long regulatory claim.",
                        "citations": [
                            {"doc_id": "bad id", "page_start": 2, "page_end": 1}
                        ],
                    }
                ],
            }
        ),
    ],
)
def test_malformed_structured_answers_fail_closed(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_structured_answer(raw)


def test_structured_claim_is_verified_as_one_unit_despite_multiple_sentences() -> None:
    structured = parse_structured_answer(
        payload(
            text=(
                "The board oversees internal controls. "
                "It also monitors whether those controls remain effective."
            )
        )
    )
    rendered_claims = render_structured_claims(structured)
    verifier_payload = {
        "answers_question": True,
        "answer_reason": "The claim answers the question.",
        "claims": [
            {"claim_index": 0, "supported": True, "reason": "Directly supported."}
        ],
    }
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(verifier_payload))
            )
        ]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response)
        )
    )

    report = verify_citations(
        render_structured_answer(structured),
        evidence(),
        client=client,
        model="fake-verifier",
        semantic=True,
        question="What does the board oversee?",
        claim_texts=rendered_claims,
    )

    assert report.valid
    assert len(report.claims) == 1
    assert report.citation_coverage == 1.0


def test_unknown_structured_source_still_fails_provenance() -> None:
    structured = parse_structured_answer(
        payload().replace("governance-doc", "unknown-doc")
    )

    report = verify_citations(
        render_structured_answer(structured),
        evidence(),
        claim_texts=render_structured_claims(structured),
    )

    assert not report.valid
    assert report.provenance_accuracy == 0.0


def test_core_answer_path_requests_schema_and_renders_claims(monkeypatch) -> None:
    captured: dict = {}
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload()))]
    )

    def create(**kwargs):
        captured.update(kwargs)
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(answer_module, "get_client", lambda: client)

    result = answer_module.answer_question(
        "What does the board oversee?",
        results=evidence(),
        verify=False,
        return_details=True,
    )

    assert result.verification.valid
    assert result.answer.startswith("The board oversees internal controls.")
    assert captured["response_format"] == structured_answer_response_format()
