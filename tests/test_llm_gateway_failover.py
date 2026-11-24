"""Offline coverage for configured LLM gateway failover and recovery."""

from types import SimpleNamespace

from llm.gateway import LLMGateway


class ProviderError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"fake HTTP {status_code}")


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class FakeModel:
    def __init__(self, outcomes, calls):
        self.outcomes = list(outcomes)
        self.calls = calls

    def invoke(self, messages, *, config=None):
        self.calls.append(messages)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_gateway(primary_outcomes, fallback_outcomes, *, clock=None):
    clock = clock or FakeClock()
    calls = {"primary": [], "secondary": []}
    models = {
        "primary": FakeModel(primary_outcomes, calls["primary"]),
        "secondary": FakeModel(fallback_outcomes, calls["secondary"]),
    }
    settings = SimpleNamespace(
        gemini_model="primary",
        gemini_fallback_model="secondary",
        llm_total_deadline_seconds=20,
        llm_primary_timeout_seconds=5,
        llm_timeout_seconds=5,
        llm_stream_timeout_seconds=5,
        llm_fallback_timeout_seconds=5,
        llm_fallback_stream_timeout_seconds=5,
        llm_max_attempts_per_model=1,
        llm_model_failure_threshold=1,
        llm_model_cooldown_seconds=10,
        llm_retry_initial_backoff_seconds=0,
        llm_retry_max_backoff_seconds=0,
    )
    gateway = LLMGateway(
        settings,
        lambda name, timeout: models[name],
        clock=clock,
        sleep=lambda _duration: None,
        random_value=lambda: 0,
    )
    return gateway, calls, clock


def test_rate_limited_primary_falls_back_to_secondary():
    gateway, calls, _clock = make_gateway(
        [ProviderError(429)], ["secondary response"]
    )

    response, _usage, record = gateway.invoke([("human", "hello")])

    assert response == "secondary response"
    assert len(calls["primary"]) == 1
    assert len(calls["secondary"]) == 1
    assert record["fallback_used"] is True
    assert record["final_model"] == "secondary"
    assert record["fallback_reason"] == "primary_rate_limit"


def test_primary_in_cooldown_is_skipped_until_recovery_probe():
    gateway, calls, clock = make_gateway(
        [ProviderError(429), "recovered primary"],
        ["first fallback", "cooldown fallback"],
    )

    first = gateway.invoke([("human", "first")])
    second = gateway.invoke([("human", "during cooldown")])

    assert first[0] == "first fallback"
    assert second[0] == "cooldown fallback"
    assert len(calls["primary"]) == 1
    assert len(calls["secondary"]) == 2

    clock.value = 10.0
    recovered = gateway.invoke([("human", "after cooldown")])

    assert recovered[0] == "recovered primary"
    assert len(calls["primary"]) == 2
    assert len(calls["secondary"]) == 2
    assert recovered[2]["fallback_used"] is False


def test_http_503_triggers_immediate_fallback():
    gateway, calls, _clock = make_gateway(
        [ProviderError(503)], ["available secondary"]
    )

    response, _usage, record = gateway.invoke([("human", "hello")])

    assert response == "available secondary"
    assert len(calls["primary"]) == len(calls["secondary"]) == 1
    assert record["fallback_used"] is True
    assert record["fallback_reason"] == "primary_service_unavailable"
