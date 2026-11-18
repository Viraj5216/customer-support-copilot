"""Customers type "order 7002"; the stored ID is "ORD7002". Never guess across customers."""
import pytest

from tools.order_resolver import resolve_order_reference

ORDERS = [
    {"order_id": "ORD7001", "product": "Pack", "status": "Delivered"},
    {"order_id": "ORD7002", "product": "Maker", "status": "Shipped"},
]


def resolve(order_id, orders=ORDERS):
    result = resolve_order_reference(
        {"order_id": order_id},
        orders,
        field_types={"order_id": "TEXT"},
    )
    return result["status"], [o["order_id"] for o in result.get("orders", [])]


@pytest.mark.parametrize("typed", ["7001", "ORD7001", "ord7001", "ORD-7001", "#7001", "ord 7001"])
def test_common_ways_of_typing_an_id_resolve(typed):
    assert resolve(typed) == ("FOUND", ["ORD7001"])


@pytest.mark.parametrize("typed", ["9999", "701", "001", "ORD", "8001"])
def test_unknown_or_partial_ids_do_not_match(typed):
    assert resolve(typed)[0] == "NOT_FOUND"


def test_ambiguous_bare_number_fails_closed():
    orders = ORDERS + [{"order_id": "INV7002", "product": "Other", "status": "Shipped"}]
    assert resolve("7002", orders)[0] == "NOT_FOUND"


def test_another_customers_id_never_resolves():
    assert resolve("8001", ORDERS)[0] == "NOT_FOUND"


def test_explicit_order_id_takes_priority_over_descriptive_filters():
    result = resolve_order_reference(
        {"order_id": "7002", "product": "Pack"},
        ORDERS,
        field_types={"order_id": "TEXT", "product": "TEXT"},
    )

    assert result["status"] == "FOUND"
    assert [order["order_id"] for order in result["orders"]] == ["ORD7002"]
