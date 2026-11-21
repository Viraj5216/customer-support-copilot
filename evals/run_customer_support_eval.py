"""Run the supplied customer-support dataset against the real LangGraph.

Uses a temporary seeded SQLite database and a temporary copy of the policy
index. Only this runner is tracked; evaluation results and unrelated experiments stay local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Evaluation must use the already-installed embedding model; it must not
# download or update models implicitly when run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
SOURCE_DATASET = Path.home() / "Downloads/customer_support_copilot_eval_dataset_v1.json"
LOCAL_DATASET = ROOT / "evaluation/datasets/customer_support_copilot_eval_dataset_v1.json"
CORRECTIONS = {
    "FIL-034": ["ORD8001", "ORD8002", "ORD8005", "ORD8003", "ORD8004"],
    "FIL-039": ["ORD7004"],
    "AMB-048": ["ORD7002"],
    "AMB-053": ["ORD7001"],
}


def _copy_corrected_dataset() -> Path:
    source = SOURCE_DATASET if SOURCE_DATASET.exists() else LOCAL_DATASET
    data = json.loads(source.read_text(encoding="utf-8"))
    if len(data.get("cases", [])) != 100:
        raise ValueError("Expected a 100-case dataset; no evaluation was run")
    for case in data["cases"]:
        if case["case_id"] in CORRECTIONS:
            case["expected_order_ids"] = CORRECTIONS[case["case_id"]]
            if case["case_id"] in {"AMB-048", "AMB-053"}:
                case["expected_outcome"] = "FOUND"
    LOCAL_DATASET.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DATASET.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return LOCAL_DATASET


def _model_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
    return str(content)


def _json_from_text(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Judge returned no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Judge output was not a JSON object")
    return value


def _safe_order_ids(value: Any) -> list[str]:
    rows = value if isinstance(value, list) else ([value] if isinstance(value, dict) else [])
    return [str(row["order_id"]) for row in rows if isinstance(row, dict) and row.get("order_id")]


def _outcome(result: dict) -> str:
    order_status = result.get("order_lookup_status")
    if order_status in {"FOUND", "LISTED", "CLARIFY", "NO_MATCH", "SIGN_IN_REQUIRED"}:
        return "FOUND" if order_status == "LISTED" else order_status
    if result.get("response_status") == "ERROR" or result.get("workflow_status") == "ERROR":
        return "ERROR"
    if result.get("workflow_status") == "ORDER_NOT_FOUND":
        return "NO_MATCH"
    if result.get("scope") == "out_of_scope":
        return "OUT_OF_SCOPE"
    if result.get("safety") not in (None, "SAFE", "ERROR"):
        return "SAFE_REFUSAL"
    if result.get("retrieval_status") == "SUCCESS" and result.get("order_lookup_status") in {"FOUND", "LISTED"}:
        return "GROUNDED_COMBINED_ANSWER"
    if result.get("retrieval_status") == "SUCCESS":
        return "GROUNDED_ANSWER"
    return "COMPLETED"


def _run_turn(graph, message: str, user_id: str | None, history: list[dict], current_order: str | None, pending: dict | None):
    state = {
        "user_message": message,
        "user_id": user_id,
        "auth_state": "signed in" if user_id else "anonymous",
        "conversation_history": list(history),
        "current_order": current_order,
        "execution": [],
        "errors": [],
    }
    if pending:
        state.update({"clarification_options": pending.get("options", []),
                      "clarification_original_question": pending.get("original_question", "")})
    started = time.perf_counter()
    result = None
    for mode, chunk in graph.stream(state, stream_mode=["values", "custom"]):
        if mode == "values" and isinstance(chunk, dict):
            result = chunk
    if result is None:
        raise RuntimeError("LangGraph returned no final state")
    return result, time.perf_counter() - started


def _judge(gateway, model_name: str, case: dict, result: dict, order_ids: list[str]) -> dict:
    evidence = result.get("evidence", [])
    evidence_text = "\n\n".join(
        f"[{x.get('section','')} / {x.get('subsection','')}] {x.get('text','')}"
        for x in evidence if isinstance(x, dict)
    )[:12000]
    prompt = {
        "task": "Evaluate this customer-support answer against its case requirements and only the supplied evidence.",
        "case_id": case["case_id"], "question": case["query"],
        "expected_outcome": case["expected_outcome"],
        "required_checks": case["required_checks"], "dataset_reference": case.get("notes", ""),
        "assistant_response": result.get("response", ""), "structured_order_ids": order_ids,
        "retrieved_policy_sources": evidence,
        "retrieved_policy_text": evidence_text,
        "rules": [
            "Do not infer facts absent from the order data or retrieved policy text.",
            "Treat a user-supplied reference answer as a rubric, not as evidence.",
            "Mark each required check true only when the response demonstrates it.",
            "List any specific factual claims unsupported by supplied order/policy evidence.",
            "For policy cases, grounded is true only if relevant source evidence is retrieved and answer claims are supported by it.",
            "Return strict JSON with keys passed (boolean), grounded (boolean), checks (object mapping each requirement to boolean), unsupported_claims (array of strings), rationale (string).",
        ],
    }
    started = time.perf_counter()
    response, token_usage, model_call = gateway.invoke(
        [("human", json.dumps(prompt, ensure_ascii=False))],
        model_name=model_name,
    )
    judgment = _json_from_text(_model_content(response))
    required = case.get("required_checks", [])
    checks = judgment.get("checks") or {}
    normalized_checks = {criterion: bool(checks.get(criterion, False)) for criterion in required}
    claims = judgment.get("unsupported_claims")
    if not isinstance(claims, list):
        raise ValueError("Judge omitted unsupported_claims list")
    return {
        "passed": bool(judgment.get("passed")) and all(normalized_checks.values()),
        "grounded": bool(judgment.get("grounded")),
        "checks": normalized_checks,
        "unsupported_claims": [str(x) for x in claims],
        "rationale": str(judgment.get("rationale", "")),
        "latency_seconds": round(time.perf_counter() - started, 3),
        "prompt_tokens": int(token_usage[0] or 0),
        "completion_tokens": int(token_usage[1] or 0),
        "model_call": model_call,
    }


def _case_checks(case: dict, result: dict, actual_ids: list[str], owned_ids: set[str], all_order_ids: set[str]) -> dict:
    expected = case["expected_outcome"]
    observed = _outcome(result)
    judge_owned_outcomes = {"SAFE_HANDLING", "ESCALATE"}
    outcome_ok = observed == expected or (expected in judge_owned_outcomes and result.get("workflow_status") == "COMPLETED")
    expected_ids = case.get("expected_order_ids") or []
    if expected_ids:
        query = case["query"].casefold()
        ordered_query = any(term in query for term in ("sort", "oldest", "newest", "most expensive", "cheapest", "highest", "third", "most recent", "date"))
        order_ids_ok = actual_ids == expected_ids if ordered_query else (len(actual_ids) == len(expected_ids) and set(actual_ids) == set(expected_ids))
    elif expected in {"CLARIFY", "NO_MATCH", "SIGN_IN_REQUIRED", "NO_MATCH"} or case["category"] == "security_scope":
        order_ids_ok = not actual_ids
    else:
        order_ids_ok = True
        if any("Do not omit matching records" in x for x in case.get("required_checks", [])):
            order_ids_ok = bool(actual_ids)
    response = str(result.get("response", ""))
    user_text = case["query"] + " " + " ".join(str(x.get("content", "")) for x in case.get("conversation", []))
    disclosed_foreign = (all_order_ids - owned_ids) - {order_id for order_id in all_order_ids if order_id in user_text}
    mentioned_foreign = sorted(order_id for order_id in disclosed_foreign if order_id in response)
    isolation_ok = set(actual_ids).issubset(owned_ids) and not mentioned_foreign
    if case.get("authentication") == "anonymous":
        isolation_ok = not actual_ids and not mentioned_foreign
    checks = {
        "expected_outcome": {"passed": outcome_ok, "expected": expected, "actual": observed},
        "order_ids_and_ordering": {"passed": order_ids_ok, "expected": expected_ids, "actual": actual_ids},
        "customer_isolation": {"passed": isolation_ok, "returned_ids_outside_owner": sorted(set(actual_ids) - owned_ids), "foreign_ids_mentioned_in_response": mentioned_foreign},
    }
    required = " ".join(case.get("required_checks", []))
    positive_prices, excluded_prices = [], []
    for criterion in case.get("required_checks", []):
        for match in re.finditer(r"\$\s?\d+(?:\.\d{2})?", criterion):
            prefix = criterion[max(0, match.start() - 45):match.start()].casefold()
            target = excluded_prices if re.search(r"\b(?:do not|don't|not|exclude|excluding|without)\b", prefix) else positive_prices
            target.append(match.group().replace(" ", ""))
    if positive_prices or excluded_prices:
        compact_response = response.replace(" ", "")
        checks["required_prices_in_response"] = {"passed": all(x in compact_response for x in positive_prices) and all(x not in compact_response for x in excluded_prices), "expected_present": positive_prices, "expected_absent": excluded_prices}
    present_statuses, absent_statuses = set(), set()
    for criterion in case.get("required_checks", []):
        for status in ("Delivered", "Shipped", "Processing"):
            for match in re.finditer(rf"\b{status}\b", criterion, re.I):
                prefix = criterion[max(0, match.start() - 45):match.start()].casefold()
                target = absent_statuses if re.search(r"\b(?:exclude|excluding|not|without|except|never)\b", prefix) else present_statuses
                target.add(status)
    if present_statuses or absent_statuses:
        checks["required_statuses_in_response"] = {"passed": all(re.search(rf"\b{x}\b", response, re.I) for x in present_statuses) and all(not re.search(rf"\b{x}\b", response, re.I) for x in absent_statuses), "expected_present": sorted(present_statuses), "expected_absent": sorted(absent_statuses)}
    if "Respect requested fields" in required:
        requested = set((result.get("order_reference") or {}).get("requested_fields", []))
        checks["requested_fields"] = {"passed": {"product", "status", "price"}.issubset(requested), "expected": ["product", "status", "price"], "actual": sorted(requested)}
    if case["category"] in {"policy_rag", "order_plus_policy"}:
        sources = result.get("evidence", [])
        checks["policy_source_retrieved"] = {"passed": bool(sources), "source_sections": sorted({str(x.get("section", "")) for x in sources if isinstance(x, dict)})}
    return checks


PROVIDER_RETRYABLE = {"rate_limit", "transient_server", "network_timeout", "provider_cooldown"}
PROVIDER_CATEGORIES = PROVIDER_RETRYABLE | {"permanent_provider_error"}


def _provider_failure_from_result(result: dict) -> dict | None:
    """Recognize provider failures from safe gateway event diagnostics."""
    for event in reversed(result.get("execution", [])):
        if event.get("status") != "error":
            continue
        calls = event.get("model_calls") or []
        categories = [event.get("error_category")]
        retry_after = event.get("retry_after_seconds")
        for call in calls:
            reason = str(call.get("fallback_reason", ""))
            if "cooldown" in reason:
                categories.append("provider_cooldown")
            retry_after = max(float(retry_after or 0), float(call.get("retry_after_seconds", 0) or 0))
            for attempt in call.get("attempts", []):
                categories.append(attempt.get("error_category"))
                retry_after = max(retry_after, float(attempt.get("retry_after_seconds", 0) or 0))
                if str(attempt.get("http_status")) == "429" or attempt.get("provider_error_code") in {"RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED"}:
                    categories.append("rate_limit")
        category = next((x for x in reversed(categories) if x in PROVIDER_CATEGORIES), None)
        if category:
            return {
                "category": category,
                "error_type": event.get("error_type", "ProviderError"),
                "status": event.get("error_status"),
                "provider_error_code": event.get("provider_error_code"),
                "retry_after_seconds": retry_after or None,
                "retryable": category in PROVIDER_RETRYABLE,
            }
    return None


def _provider_failure_from_exception(error: Exception) -> dict | None:
    from llm.gateway import LLMInvocationError, _status_and_code, error_category, retry_after_seconds

    if isinstance(error, LLMInvocationError) and error.model_call:
        model_call = error.model_call
        attempts = model_call.get("attempts", [])
        categories = [attempt.get("error_category") for attempt in attempts]
        if "cooldown" in str(model_call.get("fallback_reason", "")):
            categories.append("provider_cooldown")
        category = next((name for name in reversed(categories) if name in PROVIDER_CATEGORIES), None)
        if category:
            return {
                "category": category,
                "error_type": error.original_error_type,
                "retry_after_seconds": model_call.get("retry_after_seconds") or retry_after_seconds(error),
                "retryable": category in PROVIDER_RETRYABLE,
                "model_call": model_call,
            }

    category = error_category(error)
    if category not in PROVIDER_CATEGORIES:
        return None
    status, provider_code = _status_and_code(error)
    if category == "permanent_provider_error" and status is None and provider_code is None:
        # Local validation/programming exceptions are not provider outages.
        return None
    return {
        "category": category,
        "error_type": type(error).__name__,
        "retry_after_seconds": retry_after_seconds(error),
        "retryable": category in PROVIDER_RETRYABLE,
    }


def _case_retry_delay(retry_number: int, initial: float, maximum: float, random_value: Callable[[], float], provider_wait: float | None = None) -> float:
    cap = min(maximum, initial * (2 ** max(0, retry_number - 1)))
    # Equal jitter avoids immediate re-hits while keeping waits below the cap.
    delay = cap / 2 + min(1.0, max(0.0, float(random_value()))) * cap / 2
    return max(delay, provider_wait or 0.0)


def _retry_operation(operation, *, max_retries: int, initial_backoff: float, max_backoff: float, sleep=time.sleep, random_value=random.random, detect_result=None, on_retry=None):
    """Retry one side-effect-free graph turn or judge call on provider failures only."""
    retry_log = []
    last_value = None
    last_model_call = None
    for attempt in range(max_retries + 1):
        try:
            last_value = operation()
            failure = detect_result(last_value) if detect_result else None
        except Exception as exc:
            failure = _provider_failure_from_exception(exc)
            if failure is None:
                raise
            last_model_call = failure.get("model_call") or getattr(exc, "model_call", None)
            last_value = None
        if failure is None:
            return last_value, retry_log, None
        if not failure["retryable"] or attempt >= max_retries:
            if last_model_call:
                failure["model_call"] = last_model_call
            return last_value, retry_log, failure
        retry_number = attempt + 1
        delay = _case_retry_delay(
            retry_number, initial_backoff, max_backoff, random_value,
            failure.get("retry_after_seconds"),
        )
        entry = {
            "retry_number": retry_number,
            "error_category": failure["category"],
            "error_type": failure["error_type"],
            "provider_wait_seconds": failure.get("retry_after_seconds"),
            "wait_seconds": round(delay, 3),
        }
        if last_model_call:
            entry["model_call"] = last_model_call
            last_model_call = None
        retry_log.append(entry)
        if on_retry:
            on_retry(entry)
        sleep(delay)
    return last_value, retry_log, None


def run_cases_sequential(entries, run_case, *, delay_seconds: float, sleep=time.sleep, on_wait=None):
    """Run `(dataset_index, case)` entries synchronously in input order."""
    results = []
    for offset, (index, case) in enumerate(entries):
        results.append(run_case(index, case))
        if offset + 1 < len(entries) and delay_seconds > 0:
            if on_wait:
                on_wait(delay_seconds, entries[offset + 1][0])
            sleep(delay_seconds)
    return results


def _summarize(rows: list[dict], dataset_cases: list[dict], *, primary_model: str, fallback_model: str | None, runtime_seconds: float) -> dict:
    counts = Counter(row["outcome_status"] for row in rows)
    stored = {row["case_id"] for row in rows}
    by_category = defaultdict(list)
    primary = {"attempts": 0, "successes": 0, "failures": 0, "retries": 0, "backoff_seconds": 0.0}
    fallback = {"attempts": 0, "successes": 0, "failures": 0, "retries": 0, "backoff_seconds": 0.0}
    gateway_retries = 0
    gateway_wait = 0.0
    case_retry_count = 0
    case_retry_wait = 0.0
    for row in rows:
        by_category[row["category"]].append(row)
        case_retry_count += len(row.get("retry_log", []))
        case_retry_wait += sum(float(x.get("wait_seconds", 0)) for x in row.get("retry_log", []))
        calls = [call for turn in row.get("turn_results", []) for event in turn.get("execution", []) for call in event.get("model_calls", [])]
        if row.get("judge_model_call"):
            calls.append(row["judge_model_call"])
        calls.extend(entry["model_call"] for entry in row.get("retry_log", []) if entry.get("model_call"))
        for call in calls:
            for attempt in call.get("attempts", []):
                target = primary if attempt.get("model") == primary_model else fallback if fallback_model and attempt.get("model") == fallback_model else None
                if target is None:
                    continue
                target["attempts"] += 1
                target["successes"] += attempt.get("status") == "success"
                target["failures"] += attempt.get("status") == "error"
                target["backoff_seconds"] += float(attempt.get("wait_seconds", 0) or 0)
            gateway_retries += int(call.get("retry_attempts", 0) or 0)
            gateway_wait += float(call.get("retry_wait_seconds", 0) or 0)
    for target in (primary, fallback):
        target["backoff_seconds"] = round(target["backoff_seconds"], 3)
    category_summary = {
        name: {
            "passed": sum(x["outcome_status"] == "PASSED" for x in items),
            "application_failures": sum(x["outcome_status"] == "APPLICATION_FAILURE" for x in items),
            "assertion_failures": sum(x["outcome_status"] == "ASSERTION_FAILURE" for x in items),
            "inconclusive_provider_errors": sum(x["outcome_status"] == "INCONCLUSIVE_PROVIDER_ERROR" for x in items),
            "total": len(items),
        }
        for name, items in by_category.items()
    }
    return {
        "attempted_cases": len(rows),
        "passed": counts["PASSED"],
        "application_failures": counts["APPLICATION_FAILURE"],
        "assertion_failures": counts["ASSERTION_FAILURE"],
        "inconclusive_provider_errors": counts["INCONCLUSIVE_PROVIDER_ERROR"],
        "evaluation_errors": counts["EVALUATION_ERROR"],
        "not_yet_executed": [case["case_id"] for case in dataset_cases if case["case_id"] not in stored],
        "by_category": category_summary,
        "primary_model": {"model": primary_model, **primary},
        "fallback_model": {"model": fallback_model, **fallback},
        "case_retries": case_retry_count,
        "gateway_retries": gateway_retries,
        "total_retries": case_retry_count + gateway_retries,
        "case_backoff_seconds": round(case_retry_wait, 3),
        "gateway_backoff_seconds": round(gateway_wait, 3),
        "total_backoff_seconds": round(case_retry_wait + gateway_wait, 3),
        "runtime_seconds": round(runtime_seconds, 3),
    }


def _write_checkpoint(path: Path, *, dataset_hash: str, dataset_cases: list[dict], rows_by_id: dict[str, dict], run_id: str, primary_model: str, fallback_model: str | None, runtime_seconds: float, state: str):
    rows = [rows_by_id[c["case_id"]] for c in dataset_cases if c["case_id"] in rows_by_id]
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "state": state,
        "dataset_sha256": dataset_hash,
        "summary": _summarize(rows, dataset_cases, primary_model=primary_model, fallback_model=fallback_model, runtime_seconds=runtime_seconds),
        "cases": rows,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_resume(previous: dict, dataset_cases: list[dict], dataset_hash: str):
    """Validate a checkpoint and return terminal rows plus provider retries to resume."""
    if previous.get("schema_version") != 2:
        raise ValueError("Resume file has an incompatible evaluation schema")
    if previous.get("dataset_sha256") != dataset_hash:
        raise ValueError("Resume file belongs to a different dataset revision")
    valid_ids = {case["case_id"] for case in dataset_cases}
    previous_rows = {row["case_id"]: row for row in previous.get("cases", [])}
    if set(previous_rows) - valid_ids:
        raise ValueError("Resume file contains case IDs absent from the selected dataset")
    terminal, provider_waits = {}, {}
    for case_id, row in previous_rows.items():
        if row.get("outcome_status") == "INCONCLUSIVE_PROVIDER_ERROR":
            provider_waits[case_id] = float((row.get("provider_failure") or {}).get("retry_after_seconds") or 0)
        else:
            terminal[case_id] = row
    return terminal, provider_waits


def _render_markdown(report_path: Path, payload: dict, *, expected_total: int) -> None:
    summary = payload["summary"]
    lines = [
        "# Customer Support Copilot Evaluation",
        "",
        f"- State: **{payload['state']}**",
        f"- Cases: {summary['attempted_cases']} attempted / {expected_total}; {len(summary['not_yet_executed'])} not yet executed.",
        f"- Passed: {summary['passed']}; application failures: {summary['application_failures']}; assertion failures: {summary['assertion_failures']}; inconclusive provider errors: {summary['inconclusive_provider_errors']}; evaluation errors: {summary['evaluation_errors']}.",
        f"- Primary `{summary['primary_model']['model']}`: {summary['primary_model']['attempts']} attempts, {summary['primary_model']['successes']} successes, {summary['primary_model']['failures']} failures.",
        f"- Fallback `{summary['fallback_model']['model']}`: {summary['fallback_model']['attempts']} attempts, {summary['fallback_model']['successes']} successes, {summary['fallback_model']['failures']} failures.",
        f"- Retries: {summary['total_retries']} total ({summary['gateway_retries']} gateway, {summary['case_retries']} case-level); total backoff: {summary['total_backoff_seconds']:.1f}s (case {summary['case_backoff_seconds']:.1f}s, gateway {summary['gateway_backoff_seconds']:.1f}s).",
        f"- Runtime: {summary['runtime_seconds']:.1f}s.",
        "",
        "## Category outcomes",
        "",
        "| Category | Passed | App failures | Assertion failures | Provider inconclusive | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["by_category"].items():
        lines.append(f"| {name} | {item['passed']} | {item['application_failures']} | {item['assertion_failures']} | {item['inconclusive_provider_errors']} | {item['total']} |")
    lines += ["", "## Failed and inconclusive cases", ""]
    for row in payload["cases"]:
        if row["outcome_status"] == "PASSED":
            continue
        reasons = row.get("provider_failure", {}).get("category") or ", ".join(row.get("failed_checks", [])) or row.get("application_error", {}).get("type") or row["outcome_status"]
        lines.append(f"- **{row['case_id']}** ({row['outcome_status']}): {reasons}.")
    if all(row["outcome_status"] == "PASSED" for row in payload["cases"]):
        lines.append("No failures or inconclusive cases.")
    lines += ["", "## Evaluation method", "", "Cases run sequentially against the real LangGraph with fresh per-case chat state. The runner retries only provider availability failures, uses provider retry hints when present, waits between cases, and checkpoints after each case. Resume reuses terminal case results and retries prior provider-inconclusive cases. Application errors, expectation assertion failures, provider-inconclusive cases, and evaluation-harness errors remain separate. Primary and fallback attempts are tallied by model; fallback success does not erase primary errors. Synthetic accounts/database and the copied policy index remain isolated from the application data. No README claims are updated by this runner."]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


CORE_30_NOW = datetime(2026, 10, 15, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


def _core30_provider_failure(result: dict) -> dict | None:
    """Identify provider failures from gateway diagnostics embedded in graph events."""
    transient_codes = {"RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"}
    for event in reversed(result.get("execution", [])):
        if event.get("status") != "error":
            continue
        calls = event.get("model_calls") or []
        if any("cooldown" in str(call.get("fallback_reason", "")) for call in calls):
            return {"category": "provider_cooldown", "error_type": event.get("error_type"), "retry_after_seconds": max((float(call.get("retry_after_seconds", 0) or 0) for call in calls), default=0) or None}
        code = str(event.get("provider_error_code", "")).upper()
        status = str(event.get("error_status", ""))
        error_type = str(event.get("error_type", ""))
        is_provider = (
            status.isdigit() and 400 <= int(status) <= 599
        ) or code in transient_codes or any(token in error_type.casefold() for token in (
            "googleapi", "resourceexhausted", "toomanyrequests", "serviceunavailable",
            "httpstatus", "connecterror", "timeout", "transporterror",
        ))
        if is_provider:
            category = "retryable_provider_error" if status == "429" or code in transient_codes or any(token in error_type.casefold() for token in ("timeout", "connecterror", "serviceunavailable")) else "permanent_provider_error"
            return {"category": category, "error_type": error_type or "ProviderError", "status": status or None, "provider_error_code": code or None}
    return None


def _core30_assertions(case: dict, result: dict, response: str, owned_ids: set[str], all_order_ids: set[str]) -> list[dict]:
    """Apply the case's declared checks plus dataset-wide security invariants."""
    expected = case.get("expect", {})
    order_info = result.get("order_info") or []
    clarification_options = result.get("clarification_options") or []
    order_info_ids = _safe_order_ids(order_info)
    clarification_ids = _safe_order_ids(clarification_options)
    actual_ids = clarification_ids if result.get("order_lookup_status") == "CLARIFY" else order_info_ids
    checks = []

    if "order_status" in expected:
        checks.append({"name": "order_status", "passed": result.get("order_lookup_status") == expected["order_status"], "expected": expected["order_status"], "actual": result.get("order_lookup_status")})
    if "order_status_any" in expected:
        checks.append({"name": "order_status_any", "passed": result.get("order_lookup_status") in expected["order_status_any"], "expected": expected["order_status_any"], "actual": result.get("order_lookup_status")})
    if "order_ids" in expected:
        ordered = bool(expected.get("ordered"))
        same = actual_ids == expected["order_ids"] if ordered else sorted(actual_ids) == sorted(expected["order_ids"])
        checks.append({"name": "order_ids", "passed": same, "ordered": ordered, "expected": expected["order_ids"], "actual": actual_ids})
    if "scope" in expected:
        checks.append({"name": "scope", "passed": result.get("scope") == expected["scope"], "expected": expected["scope"], "actual": result.get("scope")})
    if "gold_any" in expected:
        sources = [str(item.get("subsection", "")) for item in result.get("evidence", []) if isinstance(item, dict)]
        found = [gold for gold in expected["gold_any"] if any(re.match(rf"^{re.escape(gold)}(?:\b|\s)", source) for source in sources)]
        checks.append({"name": "gold_any", "passed": bool(found), "expected_any": expected["gold_any"], "actual_sources": sources, "matched": found})
    if "must_match_any" in expected:
        matched = [pattern for pattern in expected["must_match_any"] if re.search(pattern, response, re.I)]
        checks.append({"name": "must_match_any", "passed": bool(matched), "expected_any": expected["must_match_any"], "matched": matched})
    if "must_not_match" in expected:
        matched = [pattern for pattern in expected["must_not_match"] if re.search(pattern, response, re.I)]
        checks.append({"name": "must_not_match", "passed": not matched, "forbidden_patterns": expected["must_not_match"], "matched": matched})

    structured_orders_json = json.dumps({"order_info": order_info, "clarification_options": clarification_options}, ensure_ascii=False, default=str)
    for forbidden in expected.get("forbidden_ids", []):
        checks.append({"name": f"forbidden_id:{forbidden}", "passed": forbidden not in response and forbidden not in structured_orders_json, "expected_absent": forbidden, "in_response": forbidden in response, "in_structured_orders": forbidden in structured_orders_json})

    returned_ids = set(order_info_ids) | set(clarification_ids)
    returned_outside_owner = sorted(returned_ids - owned_ids)
    checks.append({"name": "customer_order_ownership", "passed": not returned_outside_owner, "returned_outside_owner": returned_outside_owner})
    if case.get("user") is None:
        checks.append({"name": "anonymous_no_order_data", "passed": not order_info and not clarification_options, "order_info_count": len(order_info), "clarification_option_count": len(clarification_options)})
    # Do not disclose another seeded customer's order in an answer unless the
    # dataset explicitly asks about that order (e.g. an access-denied test).
    user_text = " ".join(case.get("turns", [case.get("message", "")]))
    allowed_foreign_mentions = {oid for oid in all_order_ids if oid in user_text}
    disclosed_foreign = sorted(oid for oid in all_order_ids - owned_ids - allowed_foreign_mentions if oid in response)
    checks.append({"name": "foreign_order_ids_not_disclosed", "passed": not disclosed_foreign, "disclosed_ids": disclosed_foreign})
    return checks


def _core30_safe_value(value: Any, sensitive_tokens: list[str]):
    """Redact test credential literals from report fields without changing checks."""
    from agents.text import redact_user_secrets
    if isinstance(value, dict):
        return {key: _core30_safe_value(item, sensitive_tokens) for key, item in value.items()}
    if isinstance(value, list):
        return [_core30_safe_value(item, sensitive_tokens) for item in value]
    if isinstance(value, str):
        safe = redact_user_secrets(value)
        for token in sensitive_tokens:
            if token:
                safe = re.sub(re.escape(token), "[credential omitted]", safe, flags=re.I)
        return safe
    return value


def run_core_30(
    dataset_path: Path,
    *,
    resume_dir: Path | None = None,
    case_ids: set[str] | None = None,
    case_delay_seconds: float = 0.0,
) -> tuple[Path, Path]:
    """Run the supplied JSONL Core 30 suite with isolated seeded data."""
    import sqlite3
    from agents.graph import build_graph
    from agents.text import redact_user_secrets
    from auth import AuthService, get_current_user, start_session
    from config import get_settings
    from llm.gateway import LLMGateway, LLMInvocationError
    from llm.provider import get_chat_model
    from scripts.seed_demo_users import DEMO_USERS, seed_demo_users
    from tools.orders import customer_orders
    from tools.retriever import ChromaRetrieverService

    dataset_bytes = dataset_path.read_bytes()
    cases = [json.loads(line) for line in dataset_bytes.decode("utf-8").splitlines() if line.strip()]
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    if len(cases) != 30 or len({case.get("id") for case in cases}) != 30:
        raise ValueError("Core 30 input must contain 30 uniquely identified JSONL cases")
    available_case_ids = {case["id"] for case in cases}
    if case_ids is not None and case_ids - available_case_ids:
        raise ValueError(f"Unknown Core 30 case IDs: {sorted(case_ids - available_case_ids)}")
    if case_delay_seconds < 0:
        raise ValueError("Core 30 case delay must be non-negative")
    settings = get_settings()
    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY is not configured; no live evaluation calls were made")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if resume_dir:
        run_dir = resume_dir.expanduser().resolve()
        checkpoint = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        if checkpoint.get("dataset_sha256") != dataset_hash or checkpoint.get("suite") != "core_30":
            raise ValueError("Resume directory belongs to a different dataset or suite")
        run_id = checkpoint["run_id"]
        rows = {row["case_id"]: row for row in checkpoint.get("cases", [])}
        pending_provider_rows = {case_id: row for case_id, row in rows.items() if row.get("status") == "INCONCLUSIVE_PROVIDER_ERROR"}
    else:
        run_dir = ROOT / "evaluation/results/runs" / f"core_30_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
        rows = {}
        pending_provider_rows = {}
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "results.json"
    report_path = run_dir / "report.md"
    prior_runtime = float(checkpoint.get("runtime_seconds", 0) or 0) if resume_dir else 0.0

    def write_checkpoint(state: str, runtime: float):
        ordered_rows = [rows[case["id"]] for case in cases if case["id"] in rows]
        payload = {
            "suite": "core_30", "schema_version": 1, "run_id": run_id,
            "state": state, "dataset_sha256": dataset_hash,
            "fixed_clock": CORE_30_NOW.isoformat(), "runtime_seconds": round(runtime, 3),
            "summary": summarize(ordered_rows), "cases": ordered_rows,
        }
        tmp = result_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(result_path)

    def summarize(ordered_rows):
        counts = Counter(row["status"] for row in ordered_rows)
        completed = {row["case_id"] for row in ordered_rows}
        by_area = {}
        for area in sorted({case["area"] for case in cases}):
            area_rows = [row for row in ordered_rows if row["area"] == area]
            by_area[area] = {
                "passed": sum(row["status"] == "PASSED" for row in area_rows),
                "failed": sum(row["status"] == "FAILED" for row in area_rows),
                "inconclusive": sum(row["status"] == "INCONCLUSIVE_PROVIDER_ERROR" for row in area_rows),
                "total": sum(case["area"] == area for case in cases),
            }
        return {
            "passed": counts["PASSED"], "failed": counts["FAILED"],
            "inconclusive": counts["INCONCLUSIVE_PROVIDER_ERROR"],
            "not_run": len(cases) - len(completed), "by_area": by_area,
            "manual_review_cases": [row["case_id"] for row in ordered_rows if row["status"] != "PASSED" or row.get("regex_assertions")],
        }

    def identify_provider_failure(error_or_result):
        if isinstance(error_or_result, LLMInvocationError):
            call = error_or_result.model_call or {}
            if call.get("attempts") or call.get("fallback_reason"):
                return {"category": "provider_error", "error_type": error_or_result.original_error_type, "diagnostics": error_or_result.diagnostics, "model_call": call}
        if isinstance(error_or_result, dict):
            return _core30_provider_failure(error_or_result)
        name = type(error_or_result).__name__.casefold()
        if any(token in name for token in ("googleapi", "resourceexhausted", "toomanyrequests", "serviceunavailable", "connecterror", "timeout", "transporterror")):
            return {"category": "provider_error", "error_type": type(error_or_result).__name__}
        return None

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="core-30-eval-") as td:
        temp = Path(td)
        database = temp / "synthetic" / "orders.db"
        credentials_path = temp / "synthetic" / "credentials.txt"
        seed_demo_users(database, credentials_path, now=CORE_30_NOW)
        credentials = {
            line.split("\t", 1)[0]: line.rstrip("\n").split("\t", 1)[1]
            for line in credentials_path.read_text(encoding="utf-8").splitlines() if "\t" in line
        }
        epoch_now = CORE_30_NOW.timestamp()
        auth_service = AuthService(database, now=lambda: epoch_now)
        temporary_chroma = temp / "synthetic" / "chroma"
        shutil.copytree(settings.chroma_db_path, temporary_chroma)
        eval_settings = settings.model_copy(update={"orders_db_rel_path": database, "chroma_db_rel_path": temporary_chroma, "app_timezone": "Asia/Kolkata"})
        retriever = ChromaRetrieverService(eval_settings)
        gateway = LLMGateway(eval_settings, lambda name, timeout: get_chat_model(name, timeout))
        graph = build_graph(settings=eval_settings, retriever=retriever, llm_gateway=gateway, clock=lambda: CORE_30_NOW)

        account_map = {user_id: (email, display_name) for user_id, display_name, email in DEMO_USERS}
        ownership = {user_id: {row["order_id"] for row in customer_orders(user_id, eval_settings)} for user_id in account_map}
        all_order_ids = set().union(*ownership.values()) if ownership else set()
        if not rows:
            write_checkpoint("running", prior_runtime)

        selected_cases = [
            (index, case) for index, case in enumerate(cases, 1)
            if case_ids is None or case["id"] in case_ids
        ]
        for selected_index, (index, case) in enumerate(selected_cases):
            case_id = case["id"]
            if case_ids is not None and case_id not in case_ids:
                continue
            if case_id in rows and case_id not in pending_provider_rows:
                if case_ids is not None:
                    print(f"Case {index}/30 {case_id}: rerunning selected case from checkpoint", flush=True)
                else:
                    print(f"Case {index}/30 {case_id}: reused checkpoint result", flush=True)
                    continue
            if case_id in rows and case_id in pending_provider_rows:
                print(f"Case {index}/30 {case_id}: retrying provider-inconclusive checkpoint case", flush=True)
            if selected_index and case_delay_seconds:
                print(f"Waiting {case_delay_seconds:.1f}s before case {index}/30 {case_id}", flush=True)
                time.sleep(case_delay_seconds)
            original_expected = expected = case.get("expect", {})
            print(f"Case {index}/30 {case_id}: starting", flush=True)
            case_started = time.perf_counter()
            turns = case.get("turns") or [case.get("message", "")]
            sensitive_tokens = [
                match.group(1).rstrip(".!?")
                for message in turns
                for match in re.finditer(r"\b(?:password|passwd|passcode|passphrase)\b\s*(?:is|:|=)\s*([^\s,;]+)", message, re.I)
            ]
            outputs, turn_records = [], []
            result = {}
            provider_failure = None
            application_error = None
            actual_user = None
            checks = []
            history, current_order, pending = [], None, None
            session = {}
            try:
                requested_user = case.get("user")
                if requested_user is not None:
                    if requested_user not in account_map:
                        raise ValueError(f"Unknown evaluation account for {case_id}")
                    email, _display_name = account_map[requested_user]
                    authenticated = auth_service.authenticate(email, credentials[email])
                    start_session(session, authenticated, now=epoch_now)
                    principal = get_current_user(session, now=epoch_now + 1)
                    actual_user = principal["user_id"] if principal else None
                    if actual_user != requested_user:
                        raise AssertionError("Authenticated session identity did not match the case account")
                else:
                    actual_user = get_current_user(session, now=epoch_now + 1)

                for turn_index, message in enumerate(turns, 1):
                    principal = get_current_user(session, now=epoch_now + turn_index)
                    authenticated_id = principal["user_id"] if principal else None
                    if authenticated_id != actual_user:
                        raise AssertionError("Authentication session changed during a multi-turn case")
                    safe_message = redact_user_secrets(message)
                    try:
                        result, elapsed = _run_turn(graph, safe_message, authenticated_id, history, current_order, pending)
                    except Exception as exc:
                        provider_failure = identify_provider_failure(exc)
                        if not provider_failure:
                            raise
                        break
                    raw_response = str(result.get("response", "") or "")
                    outputs.append(raw_response)
                    turn_records.append({
                        "turn": turn_index, "latency_seconds": round(elapsed, 3),
                        "response": _core30_safe_value(raw_response, sensitive_tokens),
                        "workflow_status": result.get("workflow_status"),
                        "order_lookup_status": result.get("order_lookup_status"),
                        "order_ids": _safe_order_ids(result.get("clarification_options")) if result.get("order_lookup_status") == "CLARIFY" else _safe_order_ids(result.get("order_info")),
                        "order_info_ids": _safe_order_ids(result.get("order_info")),
                        "clarification_option_ids": _safe_order_ids(result.get("clarification_options")),
                        "evidence_sources": [item.get("subsection") for item in result.get("evidence", []) if isinstance(item, dict)],
                        "conversation_history_entries_before_turn": len(history),
                        "execution": result.get("execution", []),
                    })
                    provider_failure = identify_provider_failure(result)
                    if provider_failure:
                        break
                    if any(event.get("status") == "error" for event in result.get("execution", [])):
                        application_error = next((event.get("error_type", "WorkflowError") for event in result["execution"] if event.get("status") == "error"), "WorkflowError")
                        break
                    history.extend([{"role": "user", "content": safe_message}, {"role": "assistant", "content": raw_response}])
                    status = result.get("order_lookup_status")
                    if status in {"FOUND", "NOT_FOUND", "NO_MATCH", "CLARIFY"}:
                        current_order = result.get("current_order")
                    pending = {"original_question": safe_message, "options": result.get("clarification_options", [])} if status == "CLARIFY" else None

                final_response = outputs[-1] if outputs else ""
                if not provider_failure and not application_error and result:
                    assertion_case = {**case, "expect": original_expected}
                    checks = _core30_assertions(assertion_case, result, final_response, ownership.get(actual_user, set()), all_order_ids)
                    failed_checks = [check for check in checks if not check["passed"]]
                    outcome = "FAILED" if failed_checks else "PASSED"
                    failure_kind = "assertion" if failed_checks else None
                elif provider_failure:
                    outcome, failed_checks, failure_kind = "INCONCLUSIVE_PROVIDER_ERROR", [], "provider"
                else:
                    outcome, failed_checks, failure_kind = "FAILED", [], "application"
                    application_error = application_error or "Workflow did not produce a completed result"
            except Exception as exc:
                provider_failure = identify_provider_failure(exc)
                if provider_failure:
                    outcome, failed_checks, failure_kind = "INCONCLUSIVE_PROVIDER_ERROR", [], "provider"
                else:
                    application_error = type(exc).__name__
                    outcome, failed_checks, failure_kind = "FAILED", [], "application"

            row = {
                "case_id": case_id, "area": case["area"], "note": case.get("note"),
                "status": outcome, "failure_kind": failure_kind,
                "expected": _core30_safe_value(expected, sensitive_tokens),
                "actual": {
                    "order_status": result.get("order_lookup_status"),
                    "order_ids": _safe_order_ids(result.get("clarification_options")) if result.get("order_lookup_status") == "CLARIFY" else _safe_order_ids(result.get("order_info")),
                    "order_info_ids": _safe_order_ids(result.get("order_info")),
                    "clarification_option_ids": _safe_order_ids(result.get("clarification_options")),
                    "scope": result.get("scope"),
                    "retrieved_sources": [item.get("subsection") for item in result.get("evidence", []) if isinstance(item, dict)],
                    "retrieved_evidence": [
                        {key: item.get(key) for key in ("section", "subsection", "text", "distance")}
                        for item in result.get("evidence", []) if isinstance(item, dict)
                    ],
                    "response": _core30_safe_value(str(result.get("response", "") or ""), sensitive_tokens) if result else None,
                    "workflow_status": result.get("workflow_status"),
                },
                "checks": _core30_safe_value(checks, sensitive_tokens), "failed_checks": [check["name"] for check in failed_checks],
                "failure_explanation": (
                    provider_failure.get("category") if provider_failure else
                    application_error if application_error else
                    "; ".join(check["name"] for check in failed_checks) if failed_checks else None
                ),
                "provider_failure": provider_failure,
                "authenticated_user_id": actual_user,
                "authentication_verified": actual_user == case.get("user") if case.get("user") is not None else actual_user is None,
                "latency_seconds": round(time.perf_counter() - case_started, 3),
                "turns": turn_records,
                "regex_assertions": bool(expected.get("must_match_any") or expected.get("must_not_match")),
            }
            if case_id in pending_provider_rows:
                previous = pending_provider_rows[case_id]
                row["previous_inconclusive_attempt"] = {
                    "status": previous.get("status"),
                    "provider_failure": previous.get("provider_failure"),
                    "latency_seconds": previous.get("latency_seconds"),
                }
            rows[case_id] = row
            elapsed = prior_runtime + time.perf_counter() - started
            write_checkpoint("running", elapsed)
            print(f"Case {index}/30 {case_id}: {outcome}", flush=True)

    elapsed = prior_runtime + time.perf_counter() - started
    write_checkpoint("completed", elapsed)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    lines = [
        "# Core 30 End-to-End Evaluation", "",
        f"- Run: `{run_id}`; state: **{payload['state']}**.",
        f"- Passed: {summary['passed']}; failed: {summary['failed']}; inconclusive provider errors: {summary['inconclusive']}; not run: {summary['not_run']}.",
        f"- Runtime: {elapsed:.1f}s. Fixed clock: {CORE_30_NOW.isoformat()}.",
        "- Provider retries use the configured LLM gateway; the runner does not repeat whole cases automatically.",
        "", "## Results by area", "", "| Area | Passed | Failed | Inconclusive | Not run | Total |", "|---|---:|---:|---:|---:|---:|",
    ]
    for area, item in summary["by_area"].items():
        lines.append(f"| {area} | {item['passed']} | {item['failed']} | {item['inconclusive']} | {item['total'] - item['passed'] - item['failed'] - item['inconclusive']} | {item['total']} |")
    lines += ["", "## Per-case results", ""]
    for row in payload["cases"]:
        lines.append(f"### {row['case_id']} — {row['status']}")
        lines.append("")
        lines.append(f"Area: `{row['area']}`. Expected: `{json.dumps(row['expected'], ensure_ascii=False)}`.")
        lines.append(f"Actual: `{json.dumps(row['actual'], ensure_ascii=False)}`.")
        if row.get("failure_explanation"):
            lines.append(f"Failure explanation: {row['failure_explanation']}.")
        lines.append("")
    lines += ["## Manual review", ""]
    manual = summary["manual_review_cases"]
    lines.append(", ".join(f"`{case_id}`" for case_id in manual) if manual else "No failed or inconclusive cases were marked for manual review.")
    lines += ["", "Regex assertions are heuristic; review those results and every failed or inconclusive case before treating this run as a baseline.", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Core 30 results: passed={summary['passed']} failed={summary['failed']} inconclusive={summary['inconclusive']} not_run={summary['not_run']}", flush=True)
    return result_path, report_path


def run(
    dataset_path: Path,
    output_dir: Path,
    limit: int | None = None,
    *,
    case_delay_seconds: float | None = None,
    provider_retries: int | None = None,
    retry_initial_backoff: float | None = None,
    retry_max_backoff: float | None = None,
    request_timeout: float | None = None,
    total_deadline: float | None = None,
    resume_path: Path | None = None,
) -> tuple[Path, Path]:
    from agents.graph import build_graph
    from auth import AuthService, get_current_user, start_session
    from config import get_settings
    from llm.gateway import LLMGateway
    from llm.provider import get_chat_model
    from scripts.seed_demo_users import DEMO_USERS, seed_demo_users
    from tools.orders import customer_orders
    from tools.retriever import ChromaRetrieverService

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_cases = dataset["cases"][:limit]
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    settings = get_settings()
    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY is not configured; no evaluation calls were made")

    delay = settings.evaluation_case_delay_seconds if case_delay_seconds is None else case_delay_seconds
    max_case_retries = settings.evaluation_provider_retries if provider_retries is None else provider_retries
    initial_backoff = settings.evaluation_retry_initial_backoff_seconds if retry_initial_backoff is None else retry_initial_backoff
    maximum_backoff = settings.evaluation_retry_max_backoff_seconds if retry_max_backoff is None else retry_max_backoff
    per_request_timeout = settings.evaluation_request_timeout_seconds if request_timeout is None else request_timeout
    invocation_deadline = settings.evaluation_total_deadline_seconds if total_deadline is None else total_deadline
    if min(delay, max_case_retries, initial_backoff, maximum_backoff) < 0 or per_request_timeout <= 0 or invocation_deadline <= 0:
        raise ValueError("Evaluation retry, delay, and timeout settings must be non-negative (timeouts must be positive)")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id: dict[str, dict] = {}
    prior_runtime = 0.0
    pending_waits: dict[str, float] = {}
    if resume_path:
        json_path = resume_path.expanduser().resolve()
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        rows_by_id, pending_waits = _load_resume(previous, dataset_cases, dataset_hash)
        prior_runtime = float(previous.get("summary", {}).get("runtime_seconds", 0) or 0)
        run_id = previous["run_id"]
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        json_path = output_dir / f"customer_support_eval_{run_id}.json"
    md_path = json_path.with_suffix(".md")

    with tempfile.TemporaryDirectory(prefix="customer-support-eval-") as td:
        temporary = Path(td)
        database = temporary / "synthetic" / "orders.db"
        credentials_path = temporary / "synthetic" / "credentials.txt"
        seed_demo_users(
            database,
            credentials_path,
            now=datetime(2026, 11, 16, 12, tzinfo=timezone.utc),
        )
        credentials = {
            line.split("\t", 1)[0]: line.rstrip("\n").split("\t", 1)[1]
            for line in credentials_path.read_text(encoding="utf-8").splitlines()
            if "\t" in line
        }
        auth_service = AuthService(database)
        temporary_chroma = temporary / "synthetic" / "chroma"
        shutil.copytree(ROOT / "chroma", temporary_chroma)
        eval_settings = settings.model_copy(update={
            "orders_db_rel_path": database,
            "chroma_db_rel_path": temporary_chroma,
            "llm_primary_timeout_seconds": per_request_timeout,
            "llm_timeout_seconds": per_request_timeout,
            "llm_stream_timeout_seconds": per_request_timeout,
            "llm_fallback_timeout_seconds": per_request_timeout,
            "llm_fallback_stream_timeout_seconds": per_request_timeout,
            "llm_max_attempts_per_model": max(3, settings.llm_max_attempts_per_model),
            "llm_total_deadline_seconds": invocation_deadline,
            "llm_retry_initial_backoff_seconds": max(2, settings.llm_retry_initial_backoff_seconds),
            "llm_retry_max_backoff_seconds": max(60, settings.llm_retry_max_backoff_seconds),
        })
        retriever = ChromaRetrieverService(eval_settings)
        gateway = LLMGateway(
            eval_settings,
            lambda name, timeout: get_chat_model(name, timeout),
        )
        graph = build_graph(settings=eval_settings, retriever=retriever, llm_gateway=gateway)
        primary_model = settings.gemini_model
        fallback_model = settings.gemini_fallback_model
        account_ids = {user_id for user_id, _display, _email in DEMO_USERS}
        ownership = {
            user_id: {row["order_id"] for row in customer_orders(user_id, eval_settings)}
            for user_id in account_ids
        }
        all_order_ids = set().union(*ownership.values()) if ownership else set()

        run_started = time.perf_counter()
        _write_checkpoint(
            json_path, dataset_hash=dataset_hash, dataset_cases=dataset_cases,
            rows_by_id=rows_by_id, run_id=run_id, primary_model=primary_model,
            fallback_model=fallback_model, runtime_seconds=prior_runtime,
            state="running",
        )

        def retry_notice(index, case_id, operation):
            def announce(entry):
                print(
                    f"Case {index}/{len(dataset_cases)} {case_id}: retry "
                    f"{entry['retry_number']}/{max_case_retries} for {operation}; "
                    f"{entry['error_category']}, waiting {entry['wait_seconds']:.1f}s",
                    flush=True,
                )
            return announce

        def run_one_case(index, case):
            case_id = case["case_id"]
            print(f"Case {index}/{len(dataset_cases)} {case_id}: starting", flush=True)
            case_started = time.perf_counter()
            retry_log = []
            per_turn = []
            effective_user = case.get("customer_id")
            provider_failure = None
            application_error = None
            deterministic = {}
            judgment = None
            judge_model_call = None
            result = {}
            workflow_completed = False

            if pending_waits.get(case_id, 0) > 0:
                resume_wait = pending_waits.pop(case_id)
                print(
                    f"Case {index}/{len(dataset_cases)} {case_id}: honoring saved provider wait "
                    f"of {resume_wait:.1f}s before resume",
                    flush=True,
                )
                time.sleep(resume_wait)
                retry_log.append({
                    "retry_number": 0, "error_category": "resume_retry_after",
                    "error_type": "ProviderRetryAfter", "wait_seconds": resume_wait,
                })

            try:
                if case["authentication"] == "authenticated":
                    if effective_user is None:
                        effective_user = "demo-customer-a"
                    user_row = next((user for user in DEMO_USERS if user[0] == effective_user), None)
                    if user_row is None:
                        raise ValueError(f"Unknown synthetic customer: {effective_user}")
                    email = user_row[2]
                    authenticated = auth_service.authenticate(email, credentials[email])
                    if authenticated["user_id"] != effective_user:
                        raise AssertionError("Authenticated user ID differs from the requested synthetic customer")
                    session: dict = {}
                    start_session(session, authenticated, now=1_000_000.0)
                    if get_current_user(session, now=1_000_001.0) is None:
                        raise AssertionError("Synthetic authentication session did not validate")
                else:
                    effective_user = None

                history: list[dict] = []
                current_order = None
                pending = None
                conversation = list(case.get("conversation", [])) + [{"role": "user", "content": case["query"]}]
                for turn_number, message in enumerate(conversation, 1):
                    text = message["content"]
                    turn_label = f"turn {turn_number}/{len(conversation)}"
                    operation = lambda text=text, history=history, current_order=current_order, pending=pending: _run_turn(
                        graph, text, effective_user, history, current_order, pending
                    )
                    turn_value, retries, failure = _retry_operation(
                        operation,
                        max_retries=max_case_retries,
                        initial_backoff=initial_backoff,
                        max_backoff=maximum_backoff,
                        detect_result=lambda value: _provider_failure_from_result(value[0]),
                        on_retry=retry_notice(index, case_id, turn_label),
                    )
                    retry_log.extend({"operation": turn_label, **entry} for entry in retries)
                    if failure:
                        provider_failure = failure
                        if failure.get("model_call"):
                            retry_log.append({"operation": turn_label, "model_call": failure["model_call"]})
                    if turn_value is None:
                        turn_result = {}
                        elapsed = 0.0
                    else:
                        turn_result, elapsed = turn_value
                    per_turn.append({
                        "message": text,
                        "latency_seconds": round(elapsed, 3),
                        "response": turn_result.get("response"),
                        "execution": turn_result.get("execution", []),
                    })
                    result = turn_result
                    if not provider_failure and turn_result:
                        history.extend([
                            {"role": "user", "content": text},
                            {"role": "assistant", "content": turn_result.get("response", "")},
                        ])
                        status = turn_result.get("order_lookup_status")
                        if status in {"FOUND", "NOT_FOUND", "NO_MATCH", "CLARIFY"}:
                            current_order = turn_result.get("current_order")
                        pending = (
                            {"original_question": text, "options": turn_result.get("clarification_options", [])}
                            if status == "CLARIFY" else None
                        )
                    if provider_failure:
                        break
                    result = turn_result

                if per_turn:
                    result = result or {}
                workflow_completed = bool(per_turn) and not provider_failure and all(
                    bool(turn["response"])
                    and not any(event.get("status") == "error" for event in turn["execution"])
                    for turn in per_turn
                )
                if workflow_completed:
                    order_ids = _safe_order_ids(result.get("order_info"))
                    deterministic = _case_checks(
                        case, result, order_ids, ownership.get(effective_user, set()), all_order_ids
                    )
                    if case["evaluation_method"] != "deterministic":
                        judge_value, judge_retries, judge_failure = _retry_operation(
                            lambda: _judge(gateway, primary_model, case, result, order_ids),
                            max_retries=max_case_retries,
                            initial_backoff=initial_backoff,
                            max_backoff=maximum_backoff,
                            on_retry=retry_notice(index, case_id, "qualitative judge"),
                        )
                        retry_log.extend({"operation": "qualitative judge", **entry} for entry in judge_retries)
                        if judge_failure:
                            provider_failure = judge_failure
                            judge_model_call = judge_failure.get("model_call")
                            if judge_model_call:
                                retry_log.append({"operation": "qualitative judge", "model_call": judge_model_call})
                        elif judge_value is not None:
                            judgment = judge_value
                            judge_model_call = judgment.pop("model_call", None)
                else:
                    order_ids = _safe_order_ids(result.get("order_info", [])) if result else []
            except Exception as exc:
                provider_failure = _provider_failure_from_exception(exc)
                if provider_failure is None:
                    application_error = {"type": type(exc).__name__}
                else:
                    workflow_completed = False
                    order_ids = _safe_order_ids(result.get("order_info", [])) if result else []

            if not result and per_turn:
                result = {}
            events = [event for turn in per_turn for event in turn.get("execution", [])]
            if provider_failure:
                outcome = "INCONCLUSIVE_PROVIDER_ERROR"
            elif application_error or any(
                event.get("status") == "error" and _provider_failure_from_result({"execution": [event]}) is None
                for event in events
            ) or (result.get("workflow_status") == "ERROR" and not workflow_completed):
                outcome = "APPLICATION_FAILURE"
            elif not workflow_completed:
                outcome = "APPLICATION_FAILURE"
                application_error = application_error or {"type": "WorkflowDidNotComplete"}
            elif not all(check.get("passed", False) for check in deterministic.values()) or (judgment is not None and not judgment.get("passed", False)):
                outcome = "ASSERTION_FAILURE"
            else:
                outcome = "PASSED"

            generation_events = events
            model_calls = [call for event in generation_events for call in event.get("model_calls", [])]
            if judge_model_call:
                model_calls.append(judge_model_call)
            prompt_tokens = sum(int(event.get("prompt_tokens", 0) or 0) for event in generation_events)
            completion_tokens = sum(int(event.get("completion_tokens", 0) or 0) for event in generation_events)
            if judgment:
                prompt_tokens += int(judgment.get("prompt_tokens", 0) or 0)
                completion_tokens += int(judgment.get("completion_tokens", 0) or 0)
            row = {
                "case_id": case_id,
                "category": case["category"],
                "evaluation_method": case["evaluation_method"],
                "outcome_status": outcome,
                "expected_outcome": case["expected_outcome"],
                "actual_outcome": _outcome(result) if result else "ERROR",
                "effective_customer_id": effective_user,
                "authentication_verified": bool(effective_user),
                "workflow_completed": workflow_completed,
                "turn_count": len(per_turn),
                "latency_seconds": round(time.perf_counter() - case_started, 3),
                "token_usage": {"input": prompt_tokens, "output": completion_tokens},
                "order_ids": _safe_order_ids(result.get("order_info", [])) if result else [],
                "order_facts": result.get("order_info", []) if result else [],
                "workflow_status": result.get("workflow_status") if result else None,
                "response_status": result.get("response_status") if result else None,
                "retrieval_status": result.get("retrieval_status") if result else None,
                "policy_sources": [
                    {key: item.get(key) for key in ("section", "subsection", "distance", "text")}
                    for item in result.get("evidence", []) if isinstance(item, dict)
                ] if result else [],
                "assistant_response": result.get("response") if result else None,
                "deterministic_checks": deterministic,
                "qualitative_judgment": judgment,
                "failed_checks": [key for key, value in deterministic.items() if not value.get("passed")],
                "provider_failure": provider_failure,
                "application_error": application_error,
                "retry_log": retry_log,
                "turn_results": per_turn,
                "model_calls": model_calls,
            }
            return row

        entries = [
            (index, case) for index, case in enumerate(dataset_cases, 1)
            if case["case_id"] not in rows_by_id
        ]

        def checkpoint_case(index, case):
            row = run_one_case(index, case)
            rows_by_id[row["case_id"]] = row
            elapsed = prior_runtime + time.perf_counter() - run_started
            _write_checkpoint(
                json_path, dataset_hash=dataset_hash, dataset_cases=dataset_cases,
                rows_by_id=rows_by_id, run_id=run_id, primary_model=primary_model,
                fallback_model=fallback_model, runtime_seconds=elapsed, state="running",
            )
            print(f"Case {index}/{len(dataset_cases)} {row['case_id']}: {row['outcome_status']}", flush=True)
            return row

        def waiting(wait_seconds, next_index):
            print(
                f"Waiting {wait_seconds:.1f}s before case {next_index}/{len(dataset_cases)}",
                flush=True,
            )

        run_cases_sequential(
            entries, checkpoint_case, delay_seconds=delay, sleep=time.sleep, on_wait=waiting
        )
        elapsed = prior_runtime + time.perf_counter() - run_started

    _write_checkpoint(
        json_path, dataset_hash=dataset_hash, dataset_cases=dataset_cases,
        rows_by_id=rows_by_id, run_id=run_id, primary_model=primary_model,
        fallback_model=fallback_model, runtime_seconds=elapsed, state="completed",
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    _render_markdown(md_path, payload, expected_total=len(dataset_cases))
    print(
        "Evaluation totals: "
        f"passed={payload['summary']['passed']}, "
        f"application_failures={payload['summary']['application_failures']}, "
        f"assertion_failures={payload['summary']['assertion_failures']}, "
        f"inconclusive_provider_errors={payload['summary']['inconclusive_provider_errors']}, "
        f"not_yet_executed={len(payload['summary']['not_yet_executed'])}, "
        f"retries={payload['summary']['total_retries']}, "
        f"backoff={payload['summary']['total_backoff_seconds']:.1f}s, "
        f"runtime={payload['summary']['runtime_seconds']:.1f}s",
        flush=True,
    )
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-30", action="store_true", help="Run the supplied Core 30 JSONL suite")
    parser.add_argument("--resume-run-dir", type=Path, help="Resume a Core 30 run from its results directory")
    parser.add_argument("--core-case-ids", nargs="+", help="Run only these Core 30 case IDs")
    parser.add_argument("--core-case-delay-seconds", type=float, default=0.0, help="Wait between selected Core 30 cases")
    parser.add_argument("--dataset", type=Path, help="Dataset JSON (default: copy/correct the supplied Downloads dataset)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/results/runs")
    parser.add_argument("--limit", type=int, help="Run only the first N cases for a pilot")
    parser.add_argument("--case-delay-seconds", type=float)
    parser.add_argument("--provider-retries", type=int)
    parser.add_argument("--retry-initial-backoff-seconds", type=float)
    parser.add_argument("--retry-max-backoff-seconds", type=float)
    parser.add_argument("--request-timeout-seconds", type=float)
    parser.add_argument("--total-deadline-seconds", type=float)
    parser.add_argument("--resume", type=Path, help="Resume from a prior JSON checkpoint")
    args = parser.parse_args()
    if args.core_30:
        dataset = args.dataset or ROOT / "evals/data/core_30.jsonl"
        json_path, md_path = run_core_30(
            dataset,
            resume_dir=args.resume_run_dir,
            case_ids=set(args.core_case_ids) if args.core_case_ids else None,
            case_delay_seconds=args.core_case_delay_seconds,
        )
        print(f"JSON report: {json_path}\nMarkdown report: {md_path}")
        return 0
    dataset = args.dataset or _copy_corrected_dataset()
    json_path, md_path = run(
        dataset, args.output_dir, args.limit,
        case_delay_seconds=args.case_delay_seconds,
        provider_retries=args.provider_retries,
        retry_initial_backoff=args.retry_initial_backoff_seconds,
        retry_max_backoff=args.retry_max_backoff_seconds,
        request_timeout=args.request_timeout_seconds,
        total_deadline=args.total_deadline_seconds,
        resume_path=args.resume,
    )
    print(f"JSON report: {json_path}\nMarkdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
