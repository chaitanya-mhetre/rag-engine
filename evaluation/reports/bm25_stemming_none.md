# Evaluation report: kestrel_v1

- created: 2026-09-25T15:21:12+00:00  ·  git: `0ce3e30`  ·  dataset sha256: `7f88fd304af8`
- items: 60 (53 answerable)  ·  k = 5  ·  embedder: `hashing-v1`
- label: BM25 stemmer = none (issue #2)

> **Offline fake-provider baseline — not a quality claim. Embeddings come from the deterministic hashing embedder and reranking from the lexical heuristic; real-model numbers are TBD.**

## Retrieval (answerable items)

| config | hit@5 | precision@5 | recall@5 | mrr | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| keyword | 0.943 | 0.204 | 0.924 | 0.844 | 0.067 | 0.119 |
| vector | 0.830 | 0.174 | 0.792 | 0.736 | 0.591 | 1.034 |
| hybrid_rrf | 0.906 | 0.193 | 0.877 | 0.826 | 0.692 | 0.791 |
| hybrid_weighted | 0.868 | 0.185 | 0.840 | 0.816 | 0.724 | 0.817 |
| hybrid_rrf_rerank | 0.962 | 0.204 | 0.934 | 0.884 | 1.258 | 1.474 |

## Recall@5 by question type

| config | adversarial | factual | keyword | multi_hop | paraphrase |
|---|---|---|---|---|---|
| keyword | 1.000 | 0.960 | 1.000 | 0.833 | 0.800 |
| vector | 0.500 | 0.920 | 0.900 | 0.667 | 0.500 |
| hybrid_rrf | 0.500 | 0.960 | 1.000 | 0.750 | 0.700 |
| hybrid_weighted | 0.500 | 0.960 | 1.000 | 0.750 | 0.500 |
| hybrid_rrf_rerank | 1.000 | 0.960 | 1.000 | 0.750 | 0.900 |

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
