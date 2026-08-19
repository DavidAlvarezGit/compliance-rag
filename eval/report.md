# A/B Evaluation Report

## Setup

- A: hybrid RAG pipeline with enforced claim-level citation verification
- B: Baseline chat model (same model, no retrieval context)
- Questions: 50

## Summary Metrics

- Correctness (RAG / baseline): 0.950 / 0.720
- Completeness (RAG / baseline): 0.955 / 0.720
- Answerable-question RAG correctness win rate: 0.0%
- Answerable correctness (RAG / baseline): 98.6% / 100.0%
- Unsupported-question refusal (RAG / baseline): 85.7% / 0.0%
- Correct answer/refusal behavior (RAG): 96.0%
- Retrieval document recall@8: 93.5%
- Claim citation coverage: 100.0%
- Citation provenance accuracy: 100.0%
- Semantic citation support rate: 100.0%
- Verified-answer rate: 100.0%
- Avg latency RAG (s): 7.190
- Avg latency Baseline (s): 4.872
- Dataset status: curated and review-ready; not independently SME approved
- Answerable questions use an LLM judge; unsupported questions use deterministic refusal scoring

## Quick Conclusion

RAG is better than baseline on judged factual correctness.

## Examples Where RAG Is Better

### Q16
**Question:** What Basel rule in 2035 sets crypto reserve ratios?
**Reference:** The provided sources do not support a sufficiently certain answer.
**RAG:** The provided sources do not support a sufficiently certain answer.
**Baseline:** As of my last knowledge update in October 2023, there was no specific Basel rule established for 2035 that sets crypto reserve ratios. The Basel Committee on Banking Supervision (BCBS) has been working on guidelines and frameworks for the treatment of crypto-assets, but any sp...
**Judged correctness:** RAG=4/4, Baseline=0/4

### Q17
**Question:** What does the corpus prescribe for TikTok exposure limits?
**Reference:** The provided sources do not support a sufficiently certain answer.
**RAG:** The provided sources do not support a sufficiently certain answer.
**Baseline:** As of my last update in October 2023, there are no specific regulations or guidelines from banking regulators that prescribe exposure limits for TikTok or similar social media platforms. Exposure limits typically pertain to financial institutions' risk management practices reg...
**Judged correctness:** RAG=4/4, Baseline=0/4

### Q18
**Question:** What is the exact Swiss GDP forecast in these documents?
**Reference:** The provided sources do not support a sufficiently certain answer.
**RAG:** The provided sources do not support a sufficiently certain answer.
**Baseline:** I'm sorry, but I do not have access to specific documents or real-time data, including exact GDP forecasts for Switzerland or any other country. For the most accurate and up-to-date information on Swiss GDP forecasts, I recommend checking official sources such as the Swiss Fed...
**Judged correctness:** RAG=4/4, Baseline=0/4

## Examples Where Baseline Is Better

### Q21
**Question:** Quelles sont les missions de la FINMA selon la LFINMA ?
**Reference:** La FINMA assure la surveillance des marchés financiers conformément aux lois applicables afin de protéger les créanciers les investisseurs les assurés et le bon fonctionnement des marchés.
**RAG:** - La FINMA consulte toutes les unités administratives dont les tâches sont liées aux projets concernés (Source: governance-2019-ordonnance-lfinma pp. 4-4). - La FINMA mène une procédure de consultation pour les ordonnances de grande portée selon la loi sur la consultation (LCo...
**Baseline:** La FINMA (Autorité fédérale de surveillance des marchés financiers) a plusieurs missions principales selon la Loi fédérale sur l'Autorité fédérale de surveillance des marchés financiers (LFINMA). Voici les principales missions :  1. **Surveillance des marchés financiers** : La...
**Judged correctness:** RAG=2/4, Baseline=4/4
