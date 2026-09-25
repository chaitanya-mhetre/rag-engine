# Evaluation report: kestrel_v1

- created: 2026-09-25T12:57:25+00:00  ·  git: `661a723`  ·  dataset sha256: `7f88fd304af8`
- items: 60 (53 answerable)  ·  k = 5  ·  embedder: `hashing-v1`
- label: M7 generation + lexical judge (offline)

> **Offline fake-provider baseline — not a quality claim. Embeddings come from the deterministic hashing embedder and reranking from the lexical heuristic; real-model numbers are TBD.**

## Retrieval (answerable items)

| config | hit@5 | precision@5 | recall@5 | mrr | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| vector | 0.830 | 0.174 | 0.792 | 0.736 | 0.528 | 0.575 |
| hybrid_rrf | 0.906 | 0.193 | 0.877 | 0.825 | 0.624 | 0.682 |
| hybrid_rrf_rerank | 0.962 | 0.204 | 0.934 | 0.884 | 1.108 | 1.496 |

## Recall@5 by question type

| config | adversarial | factual | keyword | multi_hop | paraphrase |
|---|---|---|---|---|---|
| vector | 0.500 | 0.920 | 0.900 | 0.667 | 0.500 |
| hybrid_rrf | 0.500 | 0.960 | 1.000 | 0.750 | 0.700 |
| hybrid_rrf_rerank | 1.000 | 0.960 | 1.000 | 0.750 | 0.900 |

## Generation

| config | refusal_accuracy | false_refusal_rate | keyword_recall | citation_precision | invalid_citation_rate | quote_verified_rate | adversarial_resistance | tokens_per_query | est_cost_per_query_usd | llm_p50_ms | llm_p95_ms | faithfulness | answer_relevance | context_relevance | judge | model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vector | 0.571 | 0.472 | 0.491 | 0.849 | 0.000 | 1.000 | 1.000 | 402.300 | 0.000 | 0.214 | 0.279 | 1.000 | 0.823 | 0.226 | lexical-heuristic-v1 | fake-extractive-v1 |
| hybrid_rrf | 0.429 | 0.226 | 0.698 | 0.746 | 0.000 | 1.000 | 1.000 | 701.967 | 0.000 | 0.241 | 0.307 | 1.000 | 0.784 | 0.290 | lexical-heuristic-v1 | fake-extractive-v1 |
| hybrid_rrf_rerank | 0.429 | 0.226 | 0.717 | 0.764 | 0.000 | 1.000 | 1.000 | 624.967 | 0.000 | 0.268 | 0.337 | 1.000 | 0.784 | 0.298 | lexical-heuristic-v1 | fake-extractive-v1 |

## Worst 10 items for `hybrid_rrf_rerank` (error analysis)

- **q014** (factual) recall=0.00 mrr=0.00: What is the minimum password length?  
  top-3: benefits.html>Learning Budget, leave-policy.md>Leave Policy, benefits.html>Health Insurance
- **q037** (paraphrase) recall=0.00 mrr=0.00: Can I get my beer at a client dinner paid back?  
  top-3: leave-policy.md>Vacation, oncall-runbook.md>Error Codes, leave-policy.md>Sick Leave
- **q047** (multi_hop) recall=0.50 mrr=1.00: For a 7-hour international flight, what class can I book and what is the hotel limit abroad?  
  top-3: travel-policy.md>Hotels, travel-policy.md>Travel Insurance, product-faq.md>Warranty
- **q048** (multi_hop) recall=0.50 mrr=1.00: What must a new engineer complete in their first week, and what password length is required?  
  top-3: onboarding.md>Mandatory Training, onboarding.md>30-60-90 Plan, remote-work.txt>
- **q049** (multi_hop) recall=0.50 mrr=1.00: If an E-4471 incident is a SEV1, how fast must it be acknowledged and what is the first remediation step?  
  top-3: oncall-runbook.md>Error Codes, oncall-runbook.md>Incident Command, onboarding.md>Mandatory Training
- **q024** (factual) recall=1.00 mrr=0.25: How long does a KR-7 battery last?  
  top-3: product-faq.md>KR-7 Robot Product FAQ, product-faq.md>Warranty, product-faq.md>Specifications
- **q039** (paraphrase) recall=1.00 mrr=0.25: What happens if my work laptop gets stolen?  
  top-3: onboarding.md>New Joiner Onboarding, remote-work.txt>, onboarding.md>Day One
- **q040** (paraphrase) recall=1.00 mrr=0.33: Can the robot be used outside when it's raining?  
  top-3: product-faq.md>KR-7 Robot Product FAQ, oncall-runbook.md>Error Codes, product-faq.md>Specifications
- **q041** (paraphrase) recall=1.00 mrr=0.50: How long until an empty robot is fully charged again?  
  top-3: product-faq.md>KR-7 Robot Product FAQ, product-faq.md>Battery and Charging, product-faq.md>Firmware
- **q045** (paraphrase) recall=1.00 mrr=0.50: Can my mum and dad be covered by my medical insurance?  
  top-3: travel-policy.md>Travel Insurance, benefits.html>Health Insurance, product-faq.md>Warranty
