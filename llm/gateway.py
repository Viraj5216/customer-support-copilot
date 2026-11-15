
"""Deadline-aware Gemini gateway with fast failover and model recovery."""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs


TRANSIENT_HTTP_STATUSES = {408, 500, 502, 503, 504}
FATAL_HTTP_STATUSES = {400, 401, 403, 404}

MIN_GEMINI_TIMEOUT_SECONDS = 10.0
MODEL_COOLDOWN_SECONDS = 60.0
GEMINI_SETUP_BUFFER_SECONDS = 1.0


class LLMDeadlineExceeded(TimeoutError):
    """The configured model-call deadline was reached."""


class _OneAttemptError(Exception):
    def __init__(self, original, usage=(0, 0), emitted=False):
        self.original = original
        self.usage = usage
        self.emitted = emitted
        super().__init__(type(original).__name__)


class _SetupError(Exception):
    def __init__(self, original):
        self.original = original
        super().__init__(type(original).__name__)


class _ModelFailed(Exception):
    def __init__(self, original, disposition, emitted=False):
        self.original = original
        self.disposition = disposition
        self.emitted = emitted
        super().__init__(type(original).__name__)


class LLMInvocationError(RuntimeError):
    """Terminal model error with sanitized diagnostics and call metrics."""

    def __init__(self, error, *, usage=(0, 0), model_call=None):
        self.original_error_type = type(error).__name__
        self.usage = usage
        self.model_call = model_call
        self.diagnostics = error_diagnostics(error)
        self.disposition = classify_error(error)
        self.emitted = False
        super().__init__(self.original_error_type)


def exception_chain(error):
    seen = set()
    current = error

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _safe_scalar(value):
    if isinstance(value, (str, int)):
        return str(value)[:80]

    value = getattr(value, "value", None)
    return str(value)[:80] if isinstance(value, (str, int)) else None


def _numeric_status(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _status_and_code(error):
    status = provider_code = None

    for item in exception_chain(error):
        value = getattr(item, "code", None)

        if not isinstance(value, (str, int)):
            value = getattr(item, "status_code", None)

        if status is None and isinstance(value, (str, int)):
            status = str(value)[:32]

        value = getattr(item, "status", None)

        if value is None or isinstance(value, (int, float)):
            value = getattr(item, "error_code", None)

        text = _safe_scalar(value)

        if provider_code is None and text and not text.isdigit():
            provider_code = text

    return status, provider_code


def _is_timeout_or_connection(error):
    markers = (
        "timeout",
        "connecterror",
        "connectionerror",
        "networkerror",
        "transporterror",
    )

    for item in exception_chain(error):
        name = type(item).__name__.lower()
        module = type(item).__module__.lower()

        if isinstance(item, (TimeoutError, ConnectionError)):
            return True

        if any(marker in name for marker in markers):
            return True

        if module.startswith("httpx") and (
            "timeout" in name or "transport" in name
        ):
            return True

    return False


def _is_validation_error(error):
    for item in exception_chain(error):
        name = type(item).__name__.lower()
        module = type(item).__module__.lower()

        if isinstance(item, (ValueError, TypeError)):
            return True

        if "validationerror" in name or "outputparserexception" in name:
            return True

        if "pydantic" in module and "validation" in name:
            return True

    return False


def _is_model_unavailable(error):
    if isinstance(error, LLMDeadlineExceeded):
        return False

    status, code = _status_and_code(error)
    number = _numeric_status(status)
    normalized = (code or "").upper()

    return (
        number == 429
        or number in TRANSIENT_HTTP_STATUSES
        or normalized in {
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "INTERNAL",
            "UNKNOWN",
        }
        or _is_timeout_or_connection(error)
    )


def error_diagnostics(error):
    chain = list(exception_chain(error))
    external = next(
        (
            item
            for item in chain
            if not isinstance(
                item,
                (_OneAttemptError, LLMInvocationError),
            )
        ),
        chain[0],
    )

    result = {"error_type": type(external).__name__}
    status, code = _status_and_code(error)

    if status is not None:
        result["error_status"] = status

    if code is not None:
        result["provider_error_code"] = code

    number = _numeric_status(status)
    normalized = (code or "").upper()

    if number == 429 or normalized == "RESOURCE_EXHAUSTED":
        result["error_message"] = "Provider quota or rate limit exhausted."
    elif number in FATAL_HTTP_STATUSES:
        result["error_message"] = "Provider rejected the request or credentials."
    elif (
        number in TRANSIENT_HTTP_STATUSES
        or normalized in {"UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL"}
    ):
        result["error_message"] = "Provider reported a temporary service or timeout error."
    elif _is_timeout_or_connection(error):
        result["error_message"] = "Provider connection or request timed out."

    return result


def classify_error(error):
    status, code = _status_and_code(error)
    number = _numeric_status(status)
    normalized = (code or "").upper()

    if number == 429 or normalized == "RESOURCE_EXHAUSTED":
        return "fallback"

    if number in {502, 503, 504} or normalized in {
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
    }:
        return "fallback"

    if number in FATAL_HTTP_STATUSES:
        return "fail_fast"

    if (
        number in {408, 500}
        or normalized in {"INTERNAL", "UNKNOWN"}
        or _is_timeout_or_connection(error)
    ):
        return "retry"

    if _is_validation_error(error):
        return "fail_fast"

    return "fail_fast"


def _fallback_reason(error):
    if isinstance(error, LLMDeadlineExceeded):
        return "primary_deadline_exceeded"

    status, code = _status_and_code(error)
    normalized = (code or "").upper()
    number = _numeric_status(status)

    if number == 429 or normalized == "RESOURCE_EXHAUSTED":
        return "primary_rate_limit"

    if number in {502, 503, 504} or normalized in {
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
    }:
        return "primary_service_unavailable"

    if _is_timeout_or_connection(error):
        return "primary_timeout_or_connection_error"

    if number in TRANSIENT_HTTP_STATUSES:
        return "primary_transient_error"

    return "primary_model_failure"


def _text(chunk):
    content = getattr(chunk, "content", chunk)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, str)
            or (isinstance(part, dict) and part.get("type") == "text")
        )

    return ""


def _usage(handler, value=None):
    incoming = sum(
        int(row.get("input_tokens", 0))
        for row in handler.usage_metadata.values()
    )
    outgoing = sum(
        int(row.get("output_tokens", 0))
        for row in handler.usage_metadata.values()
    )

    metadata = getattr(value, "usage_metadata", None) or {}

    return (
        incoming or int(metadata.get("input_tokens", 0)),
        outgoing or int(metadata.get("output_tokens", 0)),
    )


class LLMGateway:
    """Manage model attempts, fast fallback, recovery, and deadlines."""

    def __init__(
        self,
        settings,
        model_factory: Callable[[str, float], Any],
        *,
        clock=time.monotonic,
        sleep=time.sleep,
        random_value=random.random,
    ):
        self.settings = settings
        self.model_factory = model_factory
        self.clock = clock
        self.sleep = sleep
        self.random_value = random_value

        self._availability_lock = threading.Lock()
        self._unavailable_until = {}

    def _model_is_available(self, name):
        with self._availability_lock:
            unavailable_until = self._unavailable_until.get(name, 0.0)

        return self.clock() >= unavailable_until

    def _mark_model_unavailable(self, name):
        with self._availability_lock:
            self._unavailable_until[name] = (
                self.clock() + MODEL_COOLDOWN_SECONDS
            )

    def _mark_model_available(self, name):
        with self._availability_lock:
            self._unavailable_until.pop(name, None)

    def _available_models(self, models):
        return [
            name
            for name in models
            if self._model_is_available(name)
        ]

    def invoke(
        self,
        messages,
        *,
        schema=None,
        model_name=None,
        config: RunnableConfig | None = None,
    ):
        return self._run(
            messages,
            schema=schema,
            model_name=model_name,
            streaming=False,
            config=config,
        )

    def stream(
        self,
        messages,
        *,
        schema=None,
        model_name=None,
        writer=None,
        config: RunnableConfig | None = None,
    ):
        return self._run(
            messages,
            schema=schema,
            model_name=model_name,
            streaming=True,
            writer=writer,
            config=config,
        )

    def _run(
        self,
        messages,
        *,
        schema,
        model_name,
        streaming,
        writer=None,
        config=None,
    ):
        primary = model_name or self.settings.gemini_model
        fallback = self.settings.gemini_fallback_model

        started = self.clock()
        deadline = started + float(
            self.settings.llm_total_deadline_seconds
        )

        if fallback == primary:
            error = ValueError(
                "Fallback model must differ from primary model."
            )
            raise self._terminal(
                error, [], (0, 0), primary, False, started, None
            ) from error

        timeout = float(
            self.settings.llm_stream_timeout_seconds
            if streaming
            else self.settings.llm_timeout_seconds
        )

        configured_models = [primary]
        if fallback:
            configured_models.append(fallback)

        models = self._available_models(configured_models)

        attempts = []
        usage = [0, 0]
        fallback_used = False
        fallback_reason = None
        last_error = None

        if primary not in models:
            fallback_reason = "primary_model_cooldown"

        if not models:
            error = LLMDeadlineExceeded(
                "All configured models are temporarily unavailable. "
                "Retry after their cooldown."
            )
            raise self._terminal(
                error,
                attempts,
                tuple(usage),
                primary,
                False,
                started,
                None,
                fallback_reason="all_models_in_cooldown",
            ) from error

        for name in models:
            is_primary = name == primary
            fallback_used = not is_primary

            if not self._model_is_available(name):
                continue

            model_deadline = None

            if is_primary and fallback:
                primary_budget = (
                    max(
                        float(self.settings.llm_primary_timeout_seconds),
                        MIN_GEMINI_TIMEOUT_SECONDS,
                    )
                    + GEMINI_SETUP_BUFFER_SECONDS
                )
                model_deadline = min(
                    deadline,
                    started + primary_budget,
                )

            try:
                value, first_token_at, _ = self._run_model(
                    name,
                    messages,
                    schema,
                    streaming,
                    writer,
                    deadline,
                    attempts,
                    usage,
                    timeout,
                    config,
                    model_deadline,
                )

                self._mark_model_available(name)

                record = self._record(
                    attempts,
                    primary,
                    name,
                    fallback_used,
                    started,
                    first_token_at,
                    fallback_reason,
                )

                return value, tuple(usage), record

            except _ModelFailed as failure:
                last_error = failure.original

                if _is_model_unavailable(failure.original):
                    self._mark_model_unavailable(name)

                if failure.emitted or failure.disposition == "fail_fast":
                    raise self._terminal(
                        failure.original,
                        attempts,
                        tuple(usage),
                        primary,
                        fallback_used,
                        started,
                        name,
                        emitted=failure.emitted,
                        fallback_reason=fallback_reason,
                    ) from failure.original

                if is_primary and fallback:
                    fallback_reason = _fallback_reason(
                        failure.original
                    )

                if self.clock() >= deadline:
                    break

        error = last_error or LLMDeadlineExceeded(
            "Total LLM deadline exceeded or no model was available."
        )

        raise self._terminal(
            error,
            attempts,
            tuple(usage),
            primary,
            fallback_used,
            started,
            attempts[-1]["model"] if attempts else None,
            fallback_reason=fallback_reason,
        ) from error

    def _run_model(
        self,
        name,
        messages,
        schema,
        streaming,
        writer,
        deadline,
        attempts,
        usage,
        timeout,
        config,
        model_deadline=None,
    ):
        max_attempts = max(
            1,
            int(self.settings.llm_max_attempts_per_model),
        )

        for number in range(1, max_attempts + 1):
            attempt_deadline = deadline

            if model_deadline is not None:
                attempt_deadline = min(
                    attempt_deadline,
                    model_deadline,
                )

            remaining = attempt_deadline - self.clock()

            if remaining < MIN_GEMINI_TIMEOUT_SECONDS:
                raise _ModelFailed(
                    LLMDeadlineExceeded(
                        "Not enough time remains for another valid Gemini request."
                    ),
                    "fallback",
                )

            # Keep a one-second buffer for setup and scheduling.
            available_request_time = max(
                MIN_GEMINI_TIMEOUT_SECONDS,
                remaining - GEMINI_SETUP_BUFFER_SECONDS,
            )
            attempt_timeout = max(
                MIN_GEMINI_TIMEOUT_SECONDS,
                min(timeout, available_request_time),
            )

            attempt_started = self.clock()

            try:
                value, token_at, used = self._one_attempt(
                    name,
                    messages,
                    schema,
                    streaming,
                    writer,
                    attempt_timeout,
                    attempt_deadline,
                    config,
                )

            except _SetupError as error:
                raise _ModelFailed(
                    error.original,
                    "fail_fast",
                ) from error.original

            except _OneAttemptError as wrapped:
                error = wrapped.original
                duration = max(
                    0.0,
                    self.clock() - attempt_started,
                )
                status, code = _status_and_code(error)

                attempts.append({
                    "attempt": number,
                    "model": name,
                    "duration_seconds": round(duration, 3),
                    "status": "error",
                    "http_status": status,
                    "provider_error_code": code,
                })

                usage[0] += wrapped.usage[0]
                usage[1] += wrapped.usage[1]

                if wrapped.emitted:
                    raise _ModelFailed(
                        error,
                        "fail_fast",
                        emitted=True,
                    ) from error

                disposition = classify_error(error)

                if disposition != "retry":
                    raise _ModelFailed(
                        error,
                        disposition,
                    ) from error

                if number >= max_attempts:
                    raise _ModelFailed(
                        error,
                        "fallback",
                    ) from error

                delay = min(
                    2.0,
                    self.random_value()
                    * min(2.0, 0.25 * (2 ** (number - 1))),
                )

                remaining = attempt_deadline - self.clock()

                if remaining < delay + MIN_GEMINI_TIMEOUT_SECONDS:
                    raise _ModelFailed(
                        LLMDeadlineExceeded(
                            "Insufficient time for another valid model attempt."
                        ),
                        "fallback",
                    ) from error

                self.sleep(delay)
                continue

            duration = max(
                0.0,
                self.clock() - attempt_started,
            )

            attempts.append({
                "attempt": number,
                "model": name,
                "duration_seconds": round(duration, 3),
                "status": "success",
                "http_status": None,
                "provider_error_code": None,
            })

            usage[0] += used[0]
            usage[1] += used[1]

            if self.clock() >= attempt_deadline:
                raise _ModelFailed(
                    LLMDeadlineExceeded(
                        "Model exceeded its allocated time budget."
                    ),
                    "fallback",
                    emitted=streaming and token_at is not None,
                )

            return value, token_at, used

        raise _ModelFailed(
            LLMDeadlineExceeded(
                "Model attempt limit exhausted."
            ),
            "fallback",
        )

    def _one_attempt(
        self,
        name,
        messages,
        schema,
        streaming,
        writer,
        timeout,
        deadline,
        config,
    ):
        handler = UsageMetadataCallbackHandler()

        try:
            model = self.model_factory(name, timeout)
            target = (
                model.with_structured_output(schema)
                if schema is not None
                else model
            )
        except Exception as error:
            raise _SetupError(error) from error

        call_config = merge_configs(
            config,
            {"callbacks": [handler]},
        )

        emitted = False
        first_token_at = None
        chunks = []

        try:
            if self.clock() >= deadline:
                raise LLMDeadlineExceeded(
                    "Model time budget expired before invocation."
                )

            if streaming:
                for chunk in target.stream(
                    messages,
                    config=call_config,
                ):
                    if self.clock() >= deadline:
                        raise LLMDeadlineExceeded(
                            "Model time budget expired during streaming."
                        )

                    token = _text(chunk)
                    if not token:
                        continue

                    if first_token_at is None:
                        first_token_at = self.clock()

                    emitted = True
                    chunks.append(token)

                    if writer is not None:
                        writer({
                            "type": "response_token",
                            "text": token,
                        })

                result = "".join(chunks)

            else:
                result = target.invoke(
                    messages,
                    config=call_config,
                )

        except Exception as error:
            raise _OneAttemptError(
                error,
                _usage(handler),
                emitted,
            ) from error

        return result, first_token_at, _usage(handler, result)

    def _record(
        self,
        attempts,
        primary,
        final_model,
        fallback_used,
        started,
        first_token_at,
        fallback_reason=None,
    ):
        counts = {}

        for row in attempts:
            name = row["model"]
            counts[name] = counts.get(name, 0) + 1

        retries = sum(
            max(0, count - 1)
            for count in counts.values()
        )

        record = {
            "model": final_model or primary,
            "attempts": list(attempts),
            "retry_attempts": int(retries),
            "fallback_used": bool(fallback_used),
            "final_model": final_model,
            "status": (
                "success"
                if attempts and attempts[-1]["status"] == "success"
                else "error"
            ),
            "duration_seconds": round(
                max(0.0, self.clock() - started),
                3,
            ),
        }

        if fallback_reason is not None:
            record["fallback_reason"] = fallback_reason

        if first_token_at is not None:
            record["time_to_first_token_seconds"] = round(
                max(0.0, first_token_at - started),
                3,
            )

        return record

    def _terminal(
        self,
        error,
        attempts,
        usage,
        primary,
        fallback_used,
        started,
        final_model,
        emitted=False,
        fallback_reason=None,
    ):
        record = self._record(
            attempts,
            primary,
            final_model,
            fallback_used,
            started,
            None,
            fallback_reason,
        )
        record["status"] = "error"

        wrapped = LLMInvocationError(
            error,
            usage=usage,
            model_call=record,
        )
        wrapped.emitted = emitted
        return wrapped
