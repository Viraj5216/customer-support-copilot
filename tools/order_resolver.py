
"""Deterministic order matching with schema-validated filters and sorting."""

from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Any

from agents.schemas import FILTER_FIELD_METADATA


OPERATORS = {
    "eq", "ne", "in", "not_in",
    "contains", "not_contains",
    "gt", "gte", "lt", "lte",
    "between", "not_between", "before", "after",
    "is_null", "is_not_null",
}

LOGICAL_OPERATORS = {"AND", "OR", "NOT"}


class InvalidOrderFilter(ValueError):
    """Invalid or unsupported order filter."""


def _text(value: Any) -> str:
    return " ".join(str(value).casefold().split()) if value is not None else ""


def _order_id(value: Any) -> str:
    return str(value).strip().upper().replace(" ", "") if value is not None else ""


def _date(value: Any) -> date | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    raw = str(value).strip()

    try:
        if len(raw) == 10:
            return date.fromisoformat(raw)

        return datetime.fromisoformat(
            raw.replace("Z", "+00:00")
        ).date()
    except (TypeError, ValueError):
        return None


def _metadata(field: str, schema: dict | None) -> dict:
    if not isinstance(field, str) or not field.strip():
        raise InvalidOrderFilter("A schema field is required.")

    config = (schema or {}).get(field)

    if isinstance(config, str):
        config = {"type": config}

    if not isinstance(config, dict) or not config.get("type"):
        raise InvalidOrderFilter(f"Unsupported field: {field}")

    return {
        **FILTER_FIELD_METADATA.get(field, {}),
        **config,
    }


def _kind(config: dict) -> str:
    """Resolve a schema type to a supported execution category."""

    kind = str(config.get("type", "")).upper()

    if any(token in kind for token in (
        "INT", "REAL", "NUM", "DEC", "FLOAT", "DOUBLE",
    )):
        return "number"

    if "BOOL" in kind:
        return "boolean"

    if "DATE" in kind or "TIME" in kind:
        return "date"

    if any(token in kind for token in ("TEXT", "CHAR", "STRING", "CLOB")):
        return "text"

    raise InvalidOrderFilter(f"Unsupported field type: {kind}")


def _normalise(value: Any, config: dict) -> Any:
    kind = _kind(config)

    if kind == "number":
        if isinstance(value, bool):
            raise InvalidOrderFilter("Invalid numeric value.")

        try:
            number = float(
                str(value).strip().replace(",", "").replace("$", "")
            )
        except (TypeError, ValueError):
            raise InvalidOrderFilter("Invalid numeric value.") from None

        if not isfinite(number):
            raise InvalidOrderFilter("Numeric values must be finite.")

        if "INT" in str(config["type"]).upper() and not number.is_integer():
            raise InvalidOrderFilter(
                "Integer fields cannot use fractional values."
            )

        return number

    if kind == "boolean":
        if isinstance(value, bool):
            return value

        normalized = _text(value)

        if normalized in {"true", "1", "yes", "y"}:
            return True

        if normalized in {"false", "0", "no", "n"}:
            return False

        raise InvalidOrderFilter("Invalid boolean value.")

    if kind == "date":
        parsed = _date(value)

        if parsed is None:
            raise InvalidOrderFilter("Invalid date value.")

        return parsed

    normalized = _text(value)
    aliases = config.get("aliases") or {}

    if not isinstance(aliases, dict):
        raise InvalidOrderFilter("Invalid field aliases.")

    return _text(aliases.get(normalized, normalized))


def _conditions(specification: Any) -> list[dict]:
    if isinstance(specification, dict):
        if "operator" in specification:
            return [specification]

        if "value" in specification:
            return [{"operator": "eq", "value": specification["value"]}]

        raise InvalidOrderFilter("Invalid filter specification.")

    if isinstance(specification, (list, tuple, set)):
        values = list(specification)

        if isinstance(specification, set):
            values.sort(key=str)

        if values and all(
            isinstance(item, dict) and "operator" in item
            for item in values
        ):
            return values

        return [{"operator": "in", "value": values}]

    return [{"operator": "eq", "value": specification}]


def _prepare_condition(
    specification: dict,
    config: dict,
) -> tuple[str, list]:
    if not isinstance(specification, dict):
        raise InvalidOrderFilter("Invalid filter condition.")

    operator = specification.get("operator")

    if not isinstance(operator, str):
        raise InvalidOrderFilter("Invalid filter operator.")

    operator = operator.casefold().strip()

    if operator == "ne" and specification.get("value") is None:
        operator = "is_not_null"
    elif operator == "eq" and specification.get("value") is None:
        operator = "is_null"

    if operator not in OPERATORS:
        raise InvalidOrderFilter(f"Unsupported operator: {operator}")

    if operator in {"is_null", "is_not_null"}:
        return operator, []

    expected = specification.get("value")

    if operator in {"in", "not_in"}:
        if not isinstance(expected, (list, tuple, set)):
            raise InvalidOrderFilter(f"'{operator}' requires a list.")

        values = list(expected)

    elif operator in {"between", "not_between"}:
        if not isinstance(expected, (list, tuple)):
            raise InvalidOrderFilter(
                f"'{operator}' requires an ordered list of two values."
            )

        values = list(expected)

    else:
        values = [expected]

    if not values or any(value is None for value in values):
        raise InvalidOrderFilter("Filter values cannot be empty or null.")

    if operator in {"between", "not_between"} and len(values) != 2:
        raise InvalidOrderFilter(
            f"'{operator}' requires exactly two values."
        )

    kind = _kind(config)

    if operator in {"contains", "not_contains"} and kind != "text":
        raise InvalidOrderFilter(
            f"'{operator}' requires a text field."
        )

    if operator in {"before", "after"} and kind != "date":
        raise InvalidOrderFilter(
            f"'{operator}' requires a date field."
        )

    if operator in {
        "gt", "gte", "lt", "lte", "between", "not_between",
    } and kind not in {"number", "date"}:
        raise InvalidOrderFilter(
            f"'{operator}' requires a numeric or date field."
        )

    values = [_normalise(value, config) for value in values]

    if kind == "text" and any(not value for value in values):
        raise InvalidOrderFilter("Text filter values cannot be empty.")

    if operator in {"between", "not_between"} and values[0] > values[1]:
        raise InvalidOrderFilter(
            "Invalid range: lower value exceeds upper value."
        )

    return operator, values


def _equal(actual: Any, expected: Any, config: dict) -> bool:
    mode = config.get("match", "exact")

    if mode not in {"exact", "partial", "contains"}:
        raise InvalidOrderFilter("Unsupported field matching mode.")

    if mode == "contains":
        return expected in actual

    if mode == "partial":
        return expected in actual or actual in expected

    return actual == expected


def _match(
    actual: Any,
    prepared: tuple[str, list],
    config: dict,
) -> bool:
    operator, values = prepared

    if operator in {"is_null", "is_not_null"}:
        is_empty = actual is None or (
            isinstance(actual, str) and not actual.strip()
        )

        return is_empty if operator == "is_null" else not is_empty

    if actual is None:
        return False

    actual = _normalise(actual, config)

    if operator == "eq":
        return _equal(actual, values[0], config)

    if operator == "ne":
        return not _equal(actual, values[0], config)

    if operator == "in":
        return any(_equal(actual, value, config) for value in values)

    if operator == "not_in":
        return not any(_equal(actual, value, config) for value in values)

    if operator == "contains":
        return any(value in actual for value in values)

    if operator == "not_contains":
        return all(value not in actual for value in values)

    if operator == "between":
        return values[0] <= actual <= values[1]

    if operator == "not_between":
        return actual < values[0] or actual > values[1]

    if operator in {"gt", "after"}:
        return actual > values[0]

    if operator == "gte":
        return actual >= values[0]

    if operator in {"lt", "before"}:
        return actual < values[0]

    if operator == "lte":
        return actual <= values[0]

    return False


def _compile_leaf(
    field: Any,
    specification: dict,
    field_types: dict | None,
) -> tuple:
    config = _metadata(field, field_types)
    prepared = _prepare_condition(specification, config)

    return ("leaf", field, config, prepared)


def _compile_expression(
    expression: Any,
    field_types: dict | None,
) -> tuple:
    if not isinstance(expression, dict):
        raise InvalidOrderFilter("Invalid logical filter expression.")

    operator = expression.get("operator")

    if not isinstance(operator, str):
        raise InvalidOrderFilter("A logical operator is required.")

    operator = operator.strip().upper()

    if operator in LOGICAL_OPERATORS:
        conditions = expression.get("conditions")

        if not isinstance(conditions, list) or not conditions:
            raise InvalidOrderFilter(
                f"'{operator}' requires a non-empty conditions list."
            )

        if operator == "NOT" and len(conditions) != 1:
            raise InvalidOrderFilter(
                "'NOT' requires exactly one condition."
            )

        children = [
            _compile_expression(condition, field_types)
            for condition in conditions
        ]

        return ("group", operator, children)

    field = expression.get("field")

    if field is None:
        raise InvalidOrderFilter(
            "A comparison condition must specify a field."
        )

    return _compile_leaf(field, expression, field_types)


def _compile_legacy_filters(
    filters: dict,
    field_types: dict | None,
) -> tuple:
    """Compile the original field-to-condition format as an AND group."""

    children = []

    for field, specification in filters.items():
        for condition in _conditions(specification):
            children.append(
                _compile_leaf(field, condition, field_types)
            )

    return ("group", "AND", children)


def _evaluate(expression: tuple, order: dict) -> bool:
    kind = expression[0]

    if kind == "leaf":
        _, field, config, prepared = expression
        return _match(order.get(field), prepared, config)

    _, operator, children = expression

    if operator == "AND":
        return all(_evaluate(child, order) for child in children)

    if operator == "OR":
        return any(_evaluate(child, order) for child in children)

    if operator == "NOT":
        return not _evaluate(children[0], order)

    return False


def _apply_filters(
    orders: list[dict],
    reference: dict,
    field_types: dict | None = None,
) -> list[dict]:
    filters = reference.get("filters") or {}

    if not isinstance(filters, dict):
        raise InvalidOrderFilter("Filters must be an object.")

    if "operator" in filters or "field" in filters:
        expression = _compile_expression(filters, field_types)
    else:
        expression = _compile_legacy_filters(filters, field_types)

    implicit = []

    for field in field_types or {}:
        if reference.get(field) is not None:
            for condition in _conditions(reference[field]):
                implicit.append(
                    _compile_leaf(field, condition, field_types)
                )

    start = reference.get("date_from")
    end = reference.get("date_to")

    if start is not None or end is not None:
        field = reference.get("date_field")

        if not isinstance(field, str) or not field.strip():
            raise InvalidOrderFilter(
                "A date field is required for date filtering."
            )

        config = _metadata(field, field_types)

        if _kind(config) != "date":
            raise InvalidOrderFilter(
                "Date boundaries require a date field."
            )

        if start is not None:
            implicit.append(_compile_leaf(
                field,
                {"operator": "gte", "value": start},
                field_types,
            ))

        if end is not None:
            implicit.append(_compile_leaf(
                field,
                {"operator": "lte", "value": end},
                field_types,
            ))

    if implicit:
        expression = (
            "group",
            "AND",
            [expression, *implicit],
        )

    return [
        order
        for order in orders
        if isinstance(order, dict) and _evaluate(expression, order)
    ]


def _sort_value(value: Any, config: dict) -> Any:
    """Normalize a database value for sorting."""

    if value is None or (
        isinstance(value, str) and not value.strip()
    ):
        raise ValueError("Missing sort value.")

    return _normalise(value, config)


def _sort_orders(
    orders: list[dict],
    reference: dict,
    field_types: dict | None,
) -> list[dict]:
    """Sort records using a validated schema field."""

    position_index = reference.get("position_index")
    sort_by = reference.get("sort_by")
    sort_order = reference.get("sort_order")

    if position_index is not None:
        if (
            not isinstance(position_index, int)
            or isinstance(position_index, bool)
            or position_index < 0
        ):
            raise InvalidOrderFilter("Invalid result position.")

        if sort_by is None or sort_order not in {"asc", "desc"}:
            raise InvalidOrderFilter(
                "A valid sort field and direction are required "
                "for position selection."
            )

    if sort_by is not None:
        if not isinstance(sort_by, str) or not sort_by.strip():
            raise InvalidOrderFilter("Invalid sort field.")

        sort_by = sort_by.strip()

    if sort_order is not None and sort_order not in {"asc", "desc"}:
        raise InvalidOrderFilter("Invalid sort direction.")

    if sort_by is None:
        if sort_order is not None:
            raise InvalidOrderFilter(
                "A sort field is required when sort_order is supplied."
            )

        return orders

    config = _metadata(sort_by, field_types)
    _kind(config)

    if sort_order is None:
        sort_order = "asc"

    valid = []
    missing = []

    for index, order in enumerate(orders):
        try:
            value = _sort_value(order.get(sort_by), config)
            valid.append((value, index, order))
        except (InvalidOrderFilter, TypeError, ValueError):
            missing.append((index, order))

    valid.sort(
        key=lambda item: item[0],
        reverse=(sort_order == "desc"),
    )

    return (
        [item[2] for item in valid]
        + [item[1] for item in missing]
    )


def _apply_position(
    orders: list[dict],
    position_index: int | None,
) -> list[dict]:
    if position_index is None:
        return orders

    if (
        not isinstance(position_index, int)
        or isinstance(position_index, bool)
        or position_index < 0
    ):
        raise InvalidOrderFilter("Invalid result position.")

    if position_index >= len(orders):
        return []

    return [orders[position_index]]


def _clarification_field(
    reference: dict,
    field_types: dict | None,
) -> str | None:
    """Choose a relevant schema field for clarification labels."""

    candidates = [
        reference.get("sort_by"),
        reference.get("date_field"),
    ]

    for field in candidates:
        if not isinstance(field, str) or not field.strip():
            continue

        try:
            config = _metadata(field, field_types)
            _kind(config)
        except InvalidOrderFilter:
            continue

        return field

    return None


def _option(
    order: dict,
    index: int,
    display_field: str | None = None,
) -> dict:
    """Build a clarification option without assuming fixed data fields."""

    identifier = order.get("order_id")
    label = f"{index}. {identifier}"

    if display_field:
        value = order.get(display_field)

        if value is not None and str(value).strip():
            label += f" · {display_field}: {value}"

    return {
        "order_id": identifier,
        "label": label,
    }


def _result(
    status: str,
    orders: list[dict] | None = None,
    **extra,
) -> dict:
    orders = orders or []

    return {
        "status": status,
        "orders": orders,
        "current_order": (
            orders[0].get("order_id") if len(orders) == 1 else None
        ),
        "clarification_options": extra.get("clarification_options", []),
        "recent_orders": extra.get("recent_orders", []),
    }


def _clarify(
    orders: list[dict],
    display_field: str | None = None,
) -> dict:
    return _result(
        "CLARIFY",
        clarification_options=[
            _option(order, index, display_field)
            for index, order in enumerate(orders[:4], 1)
        ],
    )


def resolve_order_reference(
    reference: dict,
    orders: list[dict],
    *,
    current_order: str | None = None,
    references_current_order: bool = False,
    selected_order_id: str | None = None,
    clarification_options: list[dict] | None = None,
    field_types: dict | None = None,
) -> dict:
    """Resolve an order reference using authenticated customer orders."""

    if not isinstance(reference, dict):
        return _result("INVALID_FILTER")

    orders = [
        order for order in (orders or [])
        if isinstance(order, dict) and _order_id(order.get("order_id"))
    ]

    if not orders:
        try:
            _apply_filters([], reference, field_types)
            _sort_orders([], reference, field_types)
            _apply_position([], reference.get("position_index"))
        except (InvalidOrderFilter, TypeError, ValueError):
            return _result("INVALID_FILTER")

        return _result("EMPTY")

    by_id = {
        _order_id(order["order_id"]): order
        for order in orders
    }

    displayed = [
        _order_id(option.get("order_id"))
        for option in (clarification_options or [])
        if isinstance(option, dict)
        and _order_id(option.get("order_id")) in by_id
    ]

    candidates = (
        [by_id[identifier] for identifier in displayed]
        if clarification_options is not None
        else orders
    )

    requested = _order_id(reference.get("order_id"))
    selected = _order_id(selected_order_id)

    if not selected and clarification_options is not None:
        choice = reference.get("clarification_choice")

        if isinstance(choice, int) and not isinstance(choice, bool):
            if 1 <= choice <= len(displayed):
                selected = displayed[choice - 1]

    if requested:
        if requested not in by_id:
            return _result("NOT_FOUND")

        if clarification_options is not None and requested not in displayed:
            return _clarify(candidates)

        candidates = [by_id[requested]]

    if selected:
        if selected not in by_id:
            return _result("NO_MATCH")

        if clarification_options is not None and selected not in displayed:
            return _result("NO_MATCH")

        if requested and selected != requested:
            return _result("NO_MATCH")

        candidates = [by_id[selected]]

    has_criteria = any(
        reference.get(key) is not None
        for key in (
            "product",
            "variant",
            "status",
            "date_from",
            "date_to",
            "position_index",
            "sort_by",
            "sort_order",
        )
    ) or bool(reference.get("filters")) or any(
        reference.get(key)
        for key in ("list_orders", "wants_all")
    )

    current_id = _order_id(current_order)

    if (
        references_current_order
        and current_id in by_id
        and not has_criteria
        and not requested
        and not selected
        and clarification_options is None
    ):
        return _result("FOUND", [by_id[current_id]])

    try:
        matched = _apply_filters(candidates, reference, field_types)

        matched = _sort_orders(
            matched,
            reference,
            field_types,
        )

        matched = _apply_position(
            matched,
            reference.get("position_index"),
        )

    except (InvalidOrderFilter, TypeError, ValueError):
        return _result("INVALID_FILTER")

    if not matched:
        return _result("NO_MATCH")

    if selected or requested:
        return _result("FOUND", matched[:1])

    display_field = _clarification_field(reference, field_types)

    if reference.get("needs_clarification"):
        return _clarify(matched, display_field)

    if reference.get("list_orders") or reference.get("wants_all"):
        return _result("LISTED", matched)

    if reference.get("ambiguous") or len(matched) > 1:
        return _clarify(matched, display_field)

    return _result("FOUND", matched)
