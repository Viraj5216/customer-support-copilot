# Retrieval evaluation (same policy file, different retrieval settings)

- KB: `data/company_knowledge_base.txt` (71 chunks), embedding model `all-MiniLM-L6-v2`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Questions: 25 answerable + 5 unanswerable. A hit means a gold subsection is in the top k results.
- No LLM calls. Deterministic apart from model floating-point noise.

## Results

| Configuration | hit@1 | hit@3 | hit@5 | MRR | median ms |
|---|---:|---:|---:|---:|---:|
| 1. Semantic only (current default) | 16/25 | 20/25 | 22/25 | 0.723 | 6 |
| 2. + section-aware embeddings | 17/25 | 19/25 | 21/25 | 0.739 | 6 |
| 3. + BM25 hybrid | 14/25 | 20/25 | 21/25 | 0.668 | 7 |
| 4. + reranker (no BM25) | 18/25 | 23/25 | 23/25 | 0.800 | 259 |
| 5. BM25 hybrid + reranker | 18/25 | 23/25 | 23/25 | 0.800 | 264 |
| 6. All three | 18/25 | 23/25 | 23/25 | 0.800 | 263 |

Questions where an enabled stage silently fell back to the plain ranking: 3. + BM25 hybrid (1), 5. BM25 hybrid + reranker (1), 6. All three (1).

## hit@3 by question type

| Configuration | direct | paraphrase | scenario |
|---|---:|---:|---:|
| 1. Semantic only (current default) | 6/7 | 5/9 | 9/9 |
| 2. + section-aware embeddings | 7/7 | 4/9 | 8/9 |
| 3. + BM25 hybrid | 7/7 | 5/9 | 8/9 |
| 4. + reranker (no BM25) | 7/7 | 7/9 | 9/9 |
| 5. BM25 hybrid + reranker | 7/7 | 7/9 | 9/9 |
| 6. All three | 7/7 | 7/9 | 9/9 |

## Abstention (questions the KB cannot answer)

| Configuration | correctly returned nothing | answerable questions that returned nothing |
|---|---:|---:|
| 1. Semantic only (current default) | 5/5 | 1/25 |
| 2. + section-aware embeddings | 5/5 | 1/25 |
| 3. + BM25 hybrid | 0/5 | 0/25 |
| 4. + reranker (no BM25) | 0/5 | 0/25 |
| 5. BM25 hybrid + reranker | 0/5 | 0/25 |
| 6. All three | 0/5 | 0/25 |

Note: only vector-only retrieval applies the distance cutoff. Hybrid and reranker modes deliberately skip it, so they always return chunks and cannot abstain at the retrieval layer.

## Misses (answerable questions with no gold section in the top 5)

- **1. Semantic only (current default)**: R02 (top: 3.5 Delayed Orders, 3.3 Order Processing), R05 (top: nothing), R25 (top: 8.4 Personal Information, 10.1 Purpose)
- **2. + section-aware embeddings**: R02 (top: 3.5 Delayed Orders, 3.3 Order Processing), R05 (top: nothing), R06 (top: 3.9 Delivered but Not Received, 2.3 Non-Returnable Products), R18 (top: 1.3 Damaged Products, 5.3 Manufacturing Defects)
- **3. + BM25 hybrid**: R05 (top: 1.5 Sale and Discounted Products, 9.5 Free Trials), R11 (top: 7.2 Failed Payments, 4.2 Cancellation After Payment), R14 (top: 4.2 Cancellation After Payment, 1.1 Standard Refund Eligibility), R25 (top: 8.4 Personal Information, 10.1 Purpose)
- **4. + reranker (no BM25)**: R02 (top: 3.5 Delayed Orders, 3.3 Order Processing), R05 (top: 1.5 Sale and Discounted Products, 7.2 Failed Payments)
- **5. BM25 hybrid + reranker**: R02 (top: 3.5 Delayed Orders, 3.3 Order Processing), R05 (top: 1.5 Sale and Discounted Products, 7.2 Failed Payments)
- **6. All three**: R02 (top: 3.5 Delayed Orders, 3.3 Order Processing), R05 (top: 1.5 Sale and Discounted Products, 7.2 Failed Payments)
