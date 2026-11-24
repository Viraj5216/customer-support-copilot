# Eval dataset: core_30

30 end-to-end cases across the whole workflow (orders, filtering/sorting, multi-turn,
auth, return eligibility, policy answers, policy rules, no-answer, security).

Check the dataset against the KB and the seeded DB (offline, no API key):

    python evals/validate_dataset.py

## Case schema (one JSON object per line)

| field | meaning |
|---|---|
| `id`, `area`, `note` | identifiers; `note` explains what the case is probing |
| `user` | `demo-customer-a/b/single/open`, or `null` for anonymous |
| `message` / `turns` | one message, or a list of user turns run in order in one session |
| `expect` | assertions, applied to the **last** turn |

`expect` keys (all optional):

- `order_status` / `order_status_any`: `order_lookup_status` in the final graph state
- `order_ids`: ids in `order_info` (compared as a set; list order enforced only if `ordered: true`). When `order_status` is `CLARIFY`, the ids are read from `clarification_options` instead, because that is where candidates live
- `gold_any`: at least one retrieved evidence chunk must come from one of these KB subsections
- `scope`: gatekeeper scope (for example `out_of_scope`)
- `must_match_any`: at least one regex must match the response (case-insensitive)
- `must_not_match`: no regex may match the response
- `forbidden_ids`: these order ids must not appear anywhere in the response or in `order_info`

Global invariants for every case: every id in `order_info` belongs to the case's `user`,
and anonymous users never receive order data.

## Fixed environment

Seed with `seed_demo_users(..., now=2026-10-15 12:00 Asia/Kolkata)` and inject the same
clock into `build_graph`. Date-dependent cases (F04, E01-E04, M02) only mean something
under this clock. Seeded dates are relative to the seed time, so never reuse a DB seeded
on a different day.

## Known weak spots

- `must_match_any` / `must_not_match` are keyword heuristics. On the first live run, read
  every failure by hand and fix the pattern or the case before trusting the pass rate.
- F03 and F05 assume the resolver treats `price` as unit price.

## Run Core 30 and judge saved responses

With the project environment configured and `GEMINI_API_KEY` set, validate the dataset and
run the actual workflow. Run JSON and Markdown reports are created below
`evals/results/runs/`:

```bash
python evals/validate_dataset.py
python evals/run_customer_support_eval.py --core-30 --dataset evals/data/core_30.jsonl --output-dir evals/results/runs --core-case-delay-seconds 17.5
```

To qualitatively judge selected saved responses without running the workflow again, set
`RUN_JSON` to the completed run's `results.json` path and run:

```bash
RUN_JSON="evals/results/runs/core_30_<RUN_ID>/results.json"
python evals/run_customer_support_eval.py --judge-only "$RUN_JSON" --judge-delay-seconds 1
```

Judge-only evaluates `policy`, `eligibility`, `policy_rules`, `no_answer`, and `security`
cases. Other areas are listed as skipped. It reads the response and retrieved evidence from
the saved Core 30 JSON, writes `core_30_judge_<RUN_ID>.json` and `.md` beside that run, and
never edits the original results or outcomes. Existing completed judgments are reused on
resume; provider-inconclusive cases are retried. The judge model defaults to the run's
generation model, with a warning that it may favor its own answer. To use a different model,
add `--judge-model "$JUDGE_MODEL"` after setting `JUDGE_MODEL` to the desired provider model
name. The gateway applies the configured attempt, timeout, and retry/backoff settings. In the
Markdown judge report, `security` is shown separately because evidence-groundedness is not a
meaningful quality measure for a refusal; the existing security checks and case outcomes are
unchanged.
