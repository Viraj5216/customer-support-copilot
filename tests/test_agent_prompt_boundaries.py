"""Safety boundaries in prompts and order lookup, without provider calls."""

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from agents.nodes import build_nodes
from agents.schemas import GatekeeperResult


class FakeGateway:
    def __init__(self, result):
        self.result = result
        self.messages = None

    def invoke(self, messages, **_kwargs):
        self.messages = messages
        return self.result, (0, 0), {"model": "offline-fake", "attempts": []}


def app_settings():
    return SimpleNamespace(app_timezone="Asia/Kolkata")


def test_gatekeeper_matching_prompt_excludes_stored_order_and_account_identifiers(monkeypatch):
    monkeypatch.setattr(
        "agents.nodes.order_schema",
        lambda _settings: [{"name": "product", "type": "TEXT"}],
    )
    monkeypatch.setattr(
        "agents.nodes.customer_orders",
        lambda _user_id, _settings: [{
            "order_id": "ORD-PRIVATE-7002",
            "customer_id": "account-private-41",
            "customer_email": "customer@example.test",
            "product": "AeroBrew Maker",
            "variant": "Black",
        }],
    )
    gateway = FakeGateway(GatekeeperResult(
        safety="SAFE",
        scope="order_status_lookup",
        intent="order status",
        order_reference={"sort_by": None, "sort_order": None},
    ))
    nodes = build_nodes(
        app_settings=app_settings(),
        model_name="offline-fake",
        model=None,
        gateway=gateway,
        retriever=None,
        clock=lambda: datetime(2026, 10, 15, 12, tzinfo=ZoneInfo("Asia/Kolkata")),
    )

    result = nodes["gatekeeper_agent"]({
        "user_id": "customer-private-41",
        "user_message": "Where is my AeroBrew Maker?",
        "conversation_history": [],
    }, {})
    prompt = "\n".join(str(message) for message in gateway.messages)

    assert result["gatekeeper_status"] == "SUCCESS"
    assert "AeroBrew Maker" in prompt
    assert "ORD-PRIVATE-7002" not in prompt
    assert "account-private-41" not in prompt
    assert "customer-private-41" not in prompt
    assert "customer@example.test" not in prompt


def test_naive_clock_value_is_interpreted_in_the_configured_timezone(monkeypatch):
    monkeypatch.setattr("agents.nodes.order_schema", lambda _settings: [])
    monkeypatch.setattr("agents.nodes.customer_orders", lambda *_args: [])
    gateway = FakeGateway(GatekeeperResult(
        safety="SAFE",
        scope="in_scope",
        order_reference={"sort_by": None, "sort_order": None},
    ))
    nodes = build_nodes(
        app_settings=app_settings(),
        model_name="offline-fake",
        model=None,
        gateway=gateway,
        retriever=None,
        clock=lambda: datetime(2026, 10, 14, 23, 30),
    )

    result = nodes["gatekeeper_agent"]({
        "user_id": None,
        "user_message": "What is the return policy?",
        "conversation_history": [],
    }, {})
    prompt = "\n".join(str(message) for message in gateway.messages)

    assert result["gatekeeper_status"] == "SUCCESS"
    assert "Today's date in Asia/Kolkata: 2026-10-14." in prompt


def test_malformed_gateway_output_does_not_produce_an_order_reference(monkeypatch):
    monkeypatch.setattr("agents.nodes.order_schema", lambda _settings: [])
    monkeypatch.setattr("agents.nodes.customer_orders", lambda *_args: [])
    gateway = FakeGateway({"order_reference": {"order_id": "ORD7002"}})
    nodes = build_nodes(
        app_settings=app_settings(),
        model_name="offline-fake",
        model=None,
        gateway=gateway,
        retriever=None,
        clock=lambda: datetime(2026, 10, 15, 12, tzinfo=ZoneInfo("Asia/Kolkata")),
    )

    result = nodes["gatekeeper_agent"]({
        "user_id": "demo-customer-a",
        "user_message": "Where is my order?",
        "conversation_history": [],
    }, {})

    assert result["gatekeeper_status"] == "ERROR"
    assert result["safety"] == "ERROR"
    assert result.get("order_reference") is None
