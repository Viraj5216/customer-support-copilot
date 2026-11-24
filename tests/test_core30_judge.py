"""Offline tests for judging saved Core 30 responses without graph calls."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from evals.run_customer_support_eval import run_judge_only


class FakeGateway:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages, *, model_name=None):
        self.calls.append((messages, model_name))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value, (12, 5), {"model": model_name, "attempts": []}


class ProviderUnavailable(Exception):
    status_code = 503


SETTINGS = SimpleNamespace(
    gemini_model="generation-model",
    gemini_fallback_model=None,
    llm_max_attempts_per_model=2,
    llm_retry_initial_backoff_seconds=0.25,
    llm_retry_max_backoff_seconds=2.0,
)


def valid_judgment(**overrides):
    value = {
        "passed": True,
        "grounded": True,
        "checks": {"Retrieved policy evidence should include at least one subsection from: 3.1.": True},
        "unsupported_claims": [],
        "rationale": "The response is supported by the saved policy evidence.",
    }
    value.update(overrides)
    return json.dumps(value)


def make_run(tmp_path, *, areas=("policy", "out_of_scope")):
    dataset_path = tmp_path / "core_30.jsonl"
    dataset_cases = []
    result_cases = []
    for index, area in enumerate(areas):
        case_id = f"C{index:02d}"
        expect = {"gold_any": ["3.1"]} if area == "policy" else {"scope": "out_of_scope"}
        dataset_cases.append({
            "id": case_id,
            "area": area,
            "message": f"Question for {case_id}?",
            "expect": expect,
        })
        result_cases.append({
            "case_id": case_id,
            "area": area,
            "status": "PASSED",
            "actual": {
                "response": f"Saved answer for {case_id}.",
                "retrieved_evidence": [{
                    "section": "SHIPPING POLICY",
                    "subsection": "3.1 Delivery timing",
                    "text": "Standard delivery usually takes five to seven business days.",
                }],
                "order_ids": [],
            },
        })
    raw_dataset = "".join(json.dumps(case) + "\n" for case in dataset_cases).encode()
    dataset_path.write_bytes(raw_dataset)
    run_json = tmp_path / "results.json"
    run_json.write_text(json.dumps({
        "suite": "core_30",
        "state": "completed",
        "run_id": "test-run-123",
        "dataset_sha256": hashlib.sha256(raw_dataset).hexdigest(),
        "summary": {"primary_model": {"model": "generation-model"}},
        "cases": result_cases,
    }), encoding="utf-8")
    return run_json, dataset_path


def test_judge_only_records_valid_judgment_and_skips_nonselected_areas(tmp_path, monkeypatch, capsys):
    run_json, dataset = make_run(tmp_path)
    original_bytes = run_json.read_bytes()
    gateway = FakeGateway([valid_judgment()])
    monkeypatch.setattr(
        "evals.run_customer_support_eval._run_turn",
        lambda *_args, **_kwargs: pytest.fail("judge-only must not run the graph"),
    )

    json_path, md_path = run_judge_only(
        run_json,
        dataset_path=dataset,
        gateway=gateway,
        settings=SETTINGS,
    )

    output = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_path.name == "core_30_judge_test-run-123.json"
    assert md_path.is_file()
    assert len(gateway.calls) == 1
    assert gateway.calls[0][1] == "generation-model"
    prompt = json.loads(gateway.calls[0][0][0][1])
    assert prompt["assistant_response"] == "Saved answer for C00."
    assert prompt["retrieved_policy_sources"][0]["subsection"] == "3.1 Delivery timing"
    assert run_json.read_bytes() == original_bytes
    assert output["summary"]["skipped_area_count"] == 1
    assert output["skipped_case_ids"] == ["C01"]
    assert output["cases"][0]["status"] == "PASSED"
    assert output["cases"][0]["grounded"] is True
    assert output["cases"][0]["unsupported_claims"] == []
    assert output["cases"][0]["checks"]
    assert output["cases"][0]["rationale"]
    assert "WARNING: judge model matches generation model" in capsys.readouterr().out


def test_custom_judge_model_is_used_without_same_model_warning(tmp_path, capsys):
    run_json, dataset = make_run(tmp_path, areas=("policy",))
    gateway = FakeGateway([valid_judgment()])

    run_judge_only(
        run_json,
        dataset_path=dataset,
        gateway=gateway,
        settings=SETTINGS,
        judge_model="independent-reviewer",
    )

    assert gateway.calls[0][1] == "independent-reviewer"
    assert "WARNING:" not in capsys.readouterr().out


def test_markdown_reports_security_separately_and_marks_groundedness_not_applicable(tmp_path):
    run_json, dataset = make_run(tmp_path, areas=("policy", "security"))
    gateway = FakeGateway([valid_judgment(), valid_judgment()])

    _json_path, md_path = run_judge_only(
        run_json, dataset_path=dataset, gateway=gateway, settings=SETTINGS
    )

    report = md_path.read_text(encoding="utf-8")
    assert "| policy | 1 | 0 | 0 | 1 |" in report
    assert "| security | 1 | 0 | 0 | 1 |" in report
    security_case = report.split("### C01 —", 1)[1]
    assert "Grounded: `N/A (security/refusal)`" in security_case


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        ("not valid JSON", "ValueError"),
        (valid_judgment(unsupported_claims=None), "ValueError"),
    ],
)
def test_invalid_judgment_output_is_inconclusive_not_pass_or_fail(
    tmp_path, response, expected_error
):
    run_json, dataset = make_run(tmp_path, areas=("policy",))
    gateway = FakeGateway([response])

    json_path, _ = run_judge_only(
        run_json, dataset_path=dataset, gateway=gateway, settings=SETTINGS
    )

    row = json.loads(json_path.read_text(encoding="utf-8"))["cases"][0]
    assert row["status"] == "INCONCLUSIVE_JUDGE_OUTPUT"
    assert row["judge_error"] == expected_error
    assert row["grounded"] is None
    assert row["unsupported_claims"] is None


def test_provider_failure_is_recorded_as_inconclusive(tmp_path):
    run_json, dataset = make_run(tmp_path, areas=("policy",))
    gateway = FakeGateway([ProviderUnavailable("fake service unavailable")])

    json_path, _ = run_judge_only(
        run_json, dataset_path=dataset, gateway=gateway, settings=SETTINGS
    )

    row = json.loads(json_path.read_text(encoding="utf-8"))["cases"][0]
    assert row["status"] == "INCONCLUSIVE_PROVIDER_ERROR"
    assert row["provider_failure"]["status"] == "503"
    assert row["grounded"] is None
    assert row["unsupported_claims"] is None


def test_resume_skips_cases_with_completed_judgments(tmp_path):
    run_json, dataset = make_run(tmp_path, areas=("policy", "eligibility"))
    first_gateway = FakeGateway([valid_judgment(), valid_judgment()])
    json_path, _ = run_judge_only(
        run_json, dataset_path=dataset, gateway=first_gateway, settings=SETTINGS
    )
    assert len(first_gateway.calls) == 2

    second_gateway = FakeGateway([])
    resumed_path, _ = run_judge_only(
        run_json, dataset_path=dataset, gateway=second_gateway, settings=SETTINGS
    )

    output = json.loads(resumed_path.read_text(encoding="utf-8"))
    assert resumed_path == json_path
    assert second_gateway.calls == []
    assert len(output["cases"]) == 2
    assert output["summary"]["resumed_count"] == 2
