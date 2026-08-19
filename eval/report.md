# A/B Evaluation Report

## Setup

- A: reranked RAG pipeline with enforced claim-level citation verification
- B: Baseline chat model (same model, no retrieval context)
- Questions: 20

## Summary Metrics

- Correctness (RAG / baseline): 0.850 / 0.750
- Completeness (RAG / baseline): 0.850 / 0.750
- Answerable-question RAG correctness win rate: 0.0%
- Refusal accuracy (RAG): 85.0%
- Retrieval document recall@8: 90.0%
- Claim citation coverage: 100.0%
- Citation provenance accuracy: 99.2%
- Semantic citation support rate: 98.3%
- Verified-answer rate: 80.0%
- Avg latency RAG (s): 13.169
- Avg latency Baseline (s): 5.186
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
**Baseline:** As of my last update in October 2023, there are no specific regulations or guidelines from banking regulatory bodies that prescribe exposure limits for TikTok or similar social media platforms. Exposure limits typically pertain to financial institutions' risk management practi...
**Judged correctness:** RAG=4/4, Baseline=0/4

### Q18
**Question:** What is the exact Swiss GDP forecast in these documents?
**Reference:** The provided sources do not support a sufficiently certain answer.
**RAG:** The provided sources do not support a sufficiently certain answer.
**Baseline:** I'm sorry, but I do not have access to specific documents or real-time data, including exact GDP forecasts for Switzerland or any other country. For the most accurate and up-to-date information on Swiss GDP forecasts, I recommend checking official sources such as the Swiss Fed...
**Judged correctness:** RAG=4/4, Baseline=0/4

## Examples Where Baseline Is Better

### Q2
**Question:** How should banks prepare for operational disruption and maintain resilience?
**Reference:** Banks should establish governance and operational resilience arrangements including continuity planning incident response testing and recovery of critical operations.
**RAG:** The provided sources do not support a sufficiently certain answer.
**Baseline:** Banks can prepare for operational disruption and maintain resilience through a comprehensive approach that includes the following key strategies:  1. **Risk Assessment and Management**: Conduct regular risk assessments to identify potential operational disruptions, including c...
**Judged correctness:** RAG=0/4, Baseline=4/4

### Q8
**Question:** Quelles exigences de fonds propres sont encadrées par l'OFR ?
**Reference:** L'OFR encadre les fonds propres pris en compte et les exigences de fonds propres applicables aux banques selon leurs risques.
**RAG:** Les sources fournies ne permettent pas de répondre avec certitude.
**Baseline:** L'Ordonnance sur les Fonds Propres (OFR) en Suisse encadre les exigences de fonds propres pour les banques et les institutions financières. Les principales exigences de fonds propres comprennent :  1. **Ratio de fonds propres** : Les banques doivent maintenir un certain ratio ...
**Judged correctness:** RAG=0/4, Baseline=4/4

### Q15
**Question:** Quels textes du corpus encadrent les risques de crédit et le leverage ratio ?
**Reference:** Les ordonnances FINMA dédiées encadrent respectivement les risques de crédit et le leverage ratio des banques et maisons de titres.
**RAG:** Les sources fournies ne permettent pas de répondre avec certitude.
**Baseline:** Les risques de crédit et le leverage ratio sont encadrés par plusieurs textes réglementaires au niveau international et national. Voici les principaux :  1. **Bâle III** : C'est le cadre réglementaire international élaboré par le Comité de Bâle sur le contrôle bancaire. Il int...
**Judged correctness:** RAG=0/4, Baseline=4/4
