from eval.score_ab import refused


def test_refusal_detection_supports_both_corpus_languages() -> None:
    assert refused("Les sources fournies ne permettent pas de répondre avec certitude.")
    assert refused("The provided sources do not support a sufficiently certain answer.")
    assert not refused("The answer is definitely 42.")
