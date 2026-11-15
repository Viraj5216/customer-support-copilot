"""Safe execution-event and status-update helpers for graph nodes."""

import time

from llm.gateway import LLMInvocationError, error_diagnostics


def append_event(state, event):
    return [*state.get("execution", []), event]


def append_error(state, error):
    return [*state.get("errors", []), error]


def attempted_llm_call(error: Exception) -> bool:
    if not isinstance(error, LLMInvocationError):
        return False
    return bool((error.model_call or {}).get("attempts"))


def error_usage(error: Exception):
    return getattr(error, "usage", (0, 0))


def _usage(message) -> tuple[int, int]:
    metadata = getattr(message, "usage_metadata", None) or {}
    return (
        int(metadata.get("input_tokens", 0)),
        int(metadata.get("output_tokens", 0)),
    )


def event(
    node: str,
    started: float,
    summary: str,
    message=None,
    error=None,
    usage=(0, 0),
    llm_called: bool | None = None,
    model_call: dict | None = None,
) -> dict:
    input_tokens, output_tokens = usage
    if message is not None and input_tokens == output_tokens == 0:
        input_tokens, output_tokens = _usage(message)
    record = {
        "node": node,
        "status": "error" if error else "completed",
        "summary": summary,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "llm_calls": (
            (1 if message is not None else 0) if llm_called is None else int(llm_called)
        ),
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
    }
    if error:
        record.update(error_diagnostics(error))
    if model_call is not None:
        record["model_calls"] = [model_call]
    elif getattr(error, "model_call", None) is not None:
        record["model_calls"] = [error.model_call]
    return record
