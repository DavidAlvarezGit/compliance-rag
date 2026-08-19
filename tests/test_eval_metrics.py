import csv
from pathlib import Path

from eval.score_ab import refused


def test_refusal_detection_supports_both_corpus_languages() -> None:
    assert refused("Les sources fournies ne permettent pas de répondre avec certitude.")
    assert refused("The provided sources do not support a sufficiently certain answer.")
    assert not refused("The answer is definitely 42.")


def test_answer_benchmark_contains_50_unique_questions() -> None:
    dataset = Path(__file__).parents[1] / "eval" / "answer_questions.csv"
    with dataset.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 50
    assert len({row["id"] for row in rows}) == 50
    assert len({row["question"] for row in rows}) == 50
    assert {row["is_answerable"] for row in rows} == {"0", "1"}
    assert {row["language"] for row in rows} == {"EN", "FR"}
