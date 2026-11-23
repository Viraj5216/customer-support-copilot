# Retrieval evaluation (same policy file, different retrieval settings)

- KB: `data/company_knowledge_base.txt` (71 chunks), embedding model `all-MiniLM-L6-v2`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Tuning questions: 25 answerable + 5 unanswerable. Held-out questions: 10 answerable + 5 unanswerable. A hit means a gold subsection is in the top k results.
- No LLM calls. Deterministic apart from model floating-point noise.

## Abstention threshold

The reranker minimum score is `-10.73`. It was calibrated only on `retrieval_30.jsonl`: the highest hundredth-step cutoff with zero false abstentions on the answerable questions. The held-out questions were not used to select the threshold.
On the tuning set, the reranker abstained on 2/5 unanswerable questions and falsely abstained on 0/25 answerable questions.

## Tuning set: retrieval_30

| Configuration | hit@1 | hit@3 | hit@5 | MRR | median ms |
|---|---:|---:|---:|---:|---:|
| 1. Semantic only (current default) | 16/25 | 20/25 | 22/25 | 0.723 | 6 |
| 2. + section-aware embeddings | 17/25 | 19/25 | 21/25 | 0.739 | 6 |
| 3. + BM25 hybrid | 14/25 | 20/25 | 21/25 | 0.668 | 8 |
| 4. + reranker (no BM25) | 18/25 | 23/25 | 23/25 | 0.800 | 262 |
| 5. BM25 hybrid + reranker | 18/25 | 23/25 | 23/25 | 0.800 | 360 |
| 6. All three | 18/25 | 23/25 | 23/25 | 0.800 | 365 |

Questions where an enabled stage silently fell back to the plain ranking: 3. + BM25 hybrid (1), 5. BM25 hybrid + reranker (1), 6. All three (1).

### hit@3 by question type

| Configuration | direct | paraphrase | scenario |
|---|---:|---:|---:|
| 1. Semantic only (current default) | 6/7 | 5/9 | 9/9 |
| 2. + section-aware embeddings | 7/7 | 4/9 | 8/9 |
| 3. + BM25 hybrid | 7/7 | 5/9 | 8/9 |
| 4. + reranker (no BM25) | 7/7 | 7/9 | 9/9 |
| 5. BM25 hybrid + reranker | 7/7 | 7/9 | 9/9 |
| 6. All three | 7/7 | 7/9 | 9/9 |

### Abstention

| Configuration | correctly abstained on unanswerable | false abstentions on answerable | reranker-cutoff abstentions |
|---|---:|---:|---:|
| 1. Semantic only (current default) | 5/5 | 1/25 | 0 |
| 2. + section-aware embeddings | 5/5 | 1/25 | 0 |
| 3. + BM25 hybrid | 0/5 | 0/25 | 0 |
| 4. + reranker (no BM25) | 2/5 | 0/25 | 2 |
| 5. BM25 hybrid + reranker | 2/5 | 0/25 | 2 |
| 6. All three | 2/5 | 0/25 | 2 |

## Held-out set

| Configuration | hit@1 | hit@3 | hit@5 | MRR | median ms |
|---|---:|---:|---:|---:|---:|
| 1. Semantic only (current default) | 9/10 | 10/10 | 10/10 | 0.950 | 6 |
| 2. + section-aware embeddings | 9/10 | 10/10 | 10/10 | 0.950 | 7 |
| 3. + BM25 hybrid | 9/10 | 9/10 | 9/10 | 0.912 | 8 |
| 4. + reranker (no BM25) | 10/10 | 10/10 | 10/10 | 1.000 | 271 |
| 5. BM25 hybrid + reranker | 10/10 | 10/10 | 10/10 | 1.000 | 391 |
| 6. All three | 10/10 | 10/10 | 10/10 | 1.000 | 358 |

### hit@3 by question type

| Configuration | direct | paraphrase | scenario |
|---|---:|---:|---:|
| 1. Semantic only (current default) | 2/2 | 4/4 | 4/4 |
| 2. + section-aware embeddings | 2/2 | 4/4 | 4/4 |
| 3. + BM25 hybrid | 1/2 | 4/4 | 4/4 |
| 4. + reranker (no BM25) | 2/2 | 4/4 | 4/4 |
| 5. BM25 hybrid + reranker | 2/2 | 4/4 | 4/4 |
| 6. All three | 2/2 | 4/4 | 4/4 |

### Abstention

| Configuration | correctly abstained on unanswerable | false abstentions on answerable | reranker-cutoff abstentions |
|---|---:|---:|---:|
| 1. Semantic only (current default) | 1/5 | 0/10 | 0 |
| 2. + section-aware embeddings | 1/5 | 0/10 | 0 |
| 3. + BM25 hybrid | 0/5 | 0/10 | 0 |
| 4. + reranker (no BM25) | 1/5 | 0/10 | 1 |
| 5. BM25 hybrid + reranker | 1/5 | 0/10 | 1 |
| 6. All three | 1/5 | 0/10 | 1 |

Note: vector-only retrieval keeps the distance cutoff; hybrid and reranker modes skip it. Only the reranker path applies the calibrated score cutoff.

## Tuning-set misses (answerable questions with no gold section in the top 5)

- **1. Semantic only (current default)**: R02 (top: 3.5 Delayed Orders, 3.3 Order Processing), R05 (top: nothing), R25 (top: 8.4 Personal Information, 10.1 Purpose)
- **2. + section-aware embeddings**: R02 (top: 3.5 Delayed Orders, 3.3 Order Processing), R05 (top: nothing), R06 (top: 3.9 Delivered but Not Received, 2.3 Non-Returnable Products), R18 (top: 1.3 Damaged Products, 5.3 Manufacturing Defects)
- **3. + BM25 hybrid**: R05 (top: 1.5 Sale and Discounted Products, 9.5 Free Trials), R11 (top: 7.2 Failed Payments, 4.2 Cancellation After Payment), R14 (top: 4.2 Cancellation After Payment, 1.1 Standard Refund Eligibility), R25 (top: 8.4 Personal Information, 10.1 Purpose)
- **4. + reranker (no BM25)**: R02 (top: 3.5 Delayed Orders, 3.3 Order Processing), R05 (top: 1.5 Sale and Discounted Products, 7.2 Failed Payments)
- **5. BM25 hybrid + reranker**: R02 (top: 3.5 Delayed Orders, 3.3 Order Processing), R05 (top: 1.5 Sale and Discounted Products, 7.2 Failed Payments)
- **6. All three**: R02 (top: 3.5 Delayed Orders, 3.3 Order Processing), R05 (top: 1.5 Sale and Discounted Products, 7.2 Failed Payments)
