# Evaluation report: kestrel_v1

- created: 2026-09-25T15:21:14+00:00  ·  git: `0ce3e30`  ·  dataset sha256: `7f88fd304af8`
- items: 60 (53 answerable)  ·  k = 5  ·  embedder: `hashing-v1`
- label: BM25 stemmer = snowball (issue #2)

> **Offline fake-provider baseline — not a quality claim. Embeddings come from the deterministic hashing embedder and reranking from the lexical heuristic; real-model numbers are TBD.**

## Retrieval (answerable items)

| config | hit@5 | precision@5 | recall@5 | mrr | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| keyword | 0.981 | 0.219 | 0.981 | 0.919 | 0.062 | 0.137 |
| vector | 0.830 | 0.174 | 0.792 | 0.736 | 0.656 | 0.731 |
| hybrid_rrf | 0.924 | 0.200 | 0.906 | 0.844 | 0.829 | 0.98 |
| hybrid_weighted | 0.962 | 0.211 | 0.953 | 0.859 | 0.766 | 0.869 |
| hybrid_rrf_rerank | 0.981 | 0.211 | 0.962 | 0.893 | 1.305 | 1.521 |

## Recall@5 by question type

| config | adversarial | factual | keyword | multi_hop | paraphrase |
|---|---|---|---|---|---|
| keyword | 1.000 | 1.000 | 1.000 | 1.000 | 0.900 |
| vector | 0.500 | 0.920 | 0.900 | 0.667 | 0.500 |
| hybrid_rrf | 0.500 | 1.000 | 1.000 | 0.833 | 0.700 |
| hybrid_weighted | 0.500 | 1.000 | 1.000 | 0.917 | 0.900 |
| hybrid_rrf_rerank | 1.000 | 1.000 | 1.000 | 0.833 | 0.900 |

## Worst 10 items for `hybrid_rrf_rerank` (error analysis)

- **q037** (paraphrase) recall=0.00 mrr=0.00: Can I get my beer at a client dinner paid back?  
  top-3: leave-policy.md>Vacation, oncall-runbook.md>Error Codes, leave-policy.md>Sick Leave
- **q047** (multi_hop) recall=0.50 mrr=1.00: For a 7-hour international flight, what class can I book and what is the hotel limit abroad?  
  top-3: travel-policy.md>Hotels, travel-policy.md>Travel Insurance, product-faq.md>KR-7 Robot Product FAQ
- **q048** (multi_hop) recall=0.50 mrr=1.00: What must a new engineer complete in their first week, and what password length is required?  
  top-3: onboarding.md>Mandatory Training, onboarding.md>30-60-90 Plan, remote-work.txt>
- **q024** (factual) recall=1.00 mrr=0.25: How long does a KR-7 battery last?  
  top-3: product-faq.md>KR-7 Robot Product FAQ, product-faq.md>Warranty, product-faq.md>Specifications
- **q039** (paraphrase) recall=1.00 mrr=0.25: What happens if my work laptop gets stolen?  
  top-3: onboarding.md>New Joiner Onboarding, remote-work.txt>, onboarding.md>Day One
- **q040** (paraphrase) recall=1.00 mrr=0.33: Can the robot be used outside when it's raining?  
  top-3: product-faq.md>KR-7 Robot Product FAQ, oncall-runbook.md>Error Codes, product-faq.md>Specifications
- **q014** (factual) recall=1.00 mrr=0.50: What is the minimum password length?  
  top-3: benefits.html>Learning Budget, security-policy.md>Passwords and Authentication, leave-policy.md>Leave Policy
- **q041** (paraphrase) recall=1.00 mrr=0.50: How long until an empty robot is fully charged again?  
  top-3: product-faq.md>KR-7 Robot Product FAQ, product-faq.md>Battery and Charging, product-faq.md>Firmware
- **q045** (paraphrase) recall=1.00 mrr=0.50: Can my mum and dad be covered by my medical insurance?  
  top-3: travel-policy.md>Travel Insurance, benefits.html>Health Insurance, product-faq.md>Warranty
- **q059** (adversarial) recall=1.00 mrr=0.50: Is vacation unlimited at Kestrel?  
  top-3: vendor-notes.md>Important Notice, leave-policy.md>Vacation, oncall-runbook.md>On-call Runbook
