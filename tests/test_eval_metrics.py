from __future__ import annotations

import unittest

from src.eval_metrics import (
    REFUSAL_MESSAGE_EN,
    REFUSAL_MESSAGE_FR,
    has_citation,
    is_refusal,
    keyword_recall,
)


class EvalMetricsTests(unittest.TestCase):
    def test_refusal_detection_supports_both_languages(self):
        self.assertEqual(is_refusal(REFUSAL_MESSAGE_EN), 1)
        self.assertEqual(is_refusal(REFUSAL_MESSAGE_FR), 1)

    def test_citation_regex_matches_runtime_contract(self):
        text = "Executive Summary\n(Source: Basel III Capital Requirements Framework (2019) pp.12-14)"
        self.assertEqual(has_citation(text), 1)

    def test_keyword_recall_handles_unicode(self):
        score = keyword_recall(
            "Les obligations portent sur la liquidité et la gouvernance.",
            "La gouvernance et la liquidité sont couvertes.",
        )
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
