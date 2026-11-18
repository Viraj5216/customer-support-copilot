
"""Deterministic customer-facing response builders."""

import time
import re

from agents.events import append_event as _append_event, event as _event
from agents.schemas import CopilotState


SAFE_ERROR_RESPONSE = (
    "I'm sorry, we're temporarily having trouble processing your request. "
    "Please try again in a moment. If the issue is urgent, please contact "
    "customer support."
)

SIGN_IN_REQUIRED_MESSAGE = (
    "Please sign in to your account before I can look up an order."
)

UNSAFE_RESPONSE = (
    "I'm sorry, but I can't provide private, sensitive, or unauthorized "
    "customer information. I can help with legitimate account or "
    "order-related requests within our support guidelines."
)


def unsafe_action_response(state, owned_order_ids=()):
    """Give safe next steps for action requests without weakening refusals."""
    message = str(state.get("user_message", ""))
    lowered = message.casefold()
    if re.search(
        r"\b(ignore (?:all )?(?:previous|prior|above)|pretend you are|"
        r"another customer|other customer|all customers|someone else)\b",
        lowered,
    ):
        return UNSAFE_RESPONSE

    duplicate_charge = bool(re.search(
        r"\b(duplicate charge|charged twice|double[- ]charged)\b", lowered
    ))
    cancellation = bool(re.search(r"\bcancel\b.*\b(order|it)\b", lowered))
    account_change = bool(re.search(
        r"\b(change|update|modify)\b.{0,40}\b(account|address|email|phone|payment method)\b",
        lowered,
    ))
    refund_action = bool(re.search(
        r"\b(refund me|issue (?:a )?refund|process (?:a )?refund|"
        r"approve (?:my )?refund)\b",
        lowered,
    ))
    if not (duplicate_charge or cancellation or account_change or refund_action):
        return UNSAFE_RESPONSE

    if not state.get("user_id"):
        return (
            "Please sign in to your account before I can help with this request. "
            "I can't issue refunds, cancel orders, or change account information; "
            "a human support agent must review it."
        )

    reference = state.get("order_reference") or {}
    requested_id = str(reference.get("order_id") or "").strip().upper()
    if not requested_id:
        match = re.search(r"\border\s+(?:ord[- ]?)?(\d{4,})\b", lowered)
        requested_id = match.group(1).upper() if match else ""
    if requested_id:
        normalized = re.sub(r"[^A-Z0-9]", "", requested_id)
        matches = [
            str(order_id).upper()
            for order_id in owned_order_ids
            if re.sub(r"[^A-Z0-9]", "", str(order_id).upper()) == normalized
            or (normalized.isdigit() and re.sub(r"[^A-Z0-9]", "", str(order_id).upper()).endswith(normalized))
        ]
        if len(matches) != 1:
            return UNSAFE_RESPONSE

    if duplicate_charge:
        return (
            "I can't confirm whether a duplicate charge occurred or issue a refund. "
            "Duplicate-charge reports require transaction verification, and a refund "
            "should not be promised until the duplicate transaction is confirmed. "
            "Please ask a human billing support agent to review this."
        )
    if cancellation:
        return (
            "I can't cancel an order myself. A human support agent needs to review "
            "the request, and I can't confirm a cancellation unless the workflow "
            "verifies it. Please contact customer support to request a review."
        )
    if account_change:
        return (
            "I can't make account changes myself. For security, a human support "
            "agent needs to verify your identity before reviewing an account change. "
            "Please contact customer support."
        )
    return (
        "I can't issue or approve a refund myself. Refund eligibility and requests "
        "must be reviewed under the applicable policy by a human support agent. "
        "I can't promise a refund; please contact customer support for review."
    )

OUT_OF_SCOPE_RESPONSE = (
    "Sorry, but this question is outside the scope of our customer support "
    "assistant. I can help with supported customer-service requests."
)

NO_POLICY_RESPONSE = (
    "I couldn't find relevant company information for this question. "
    "Please contact customer support for help."
)


def _fixed_response_node(
    node_name, response, response_status, workflow_status, summary,
    extra_state=None,
):
    """Build a deterministic response node and record its execution event."""

    def node(state: CopilotState) -> dict:
        started = time.perf_counter()
        updates = (
            extra_state(state)
            if callable(extra_state)
            else dict(extra_state or {})
        )
        status = (
            response_status(state)
            if callable(response_status)
            else response_status
        )

        return {
            **updates,
            "response": response(state) if callable(response) else response,
            "response_status": status,
            "workflow_status": workflow_status,
            "execution": _append_event(
                state,
                _event(node_name, started, summary, llm_called=False),
            ),
        }

    return node


def _order_option_label(item):
    if not isinstance(item, dict):
        return "Unknown order"

    label = str(item.get("label") or "")
    return label.split(". ", 1)[1] if ". " in label else label or "Unknown order"


def _field_label(field: str) -> str:
    return field.replace("_", " ").strip().title()


def _format_value(value) -> str:
    if value is None or str(value).strip() == "":
        return "Unavailable"
    return str(value)


def _order_lines(
    orders: list[dict],
    requested_fields,
) -> tuple[list[str], list[str]]:
    available_fields = list(dict.fromkeys(
        field
        for item in orders
        if isinstance(item, dict)
        for field in item
    ))

    if isinstance(requested_fields, str):
        requested_fields = [requested_fields]
    elif not isinstance(requested_fields, (list, tuple)):
        requested_fields = []

    requested_fields = list(dict.fromkeys(
        field for field in requested_fields
        if isinstance(field, str) and field
    ))

    available_set = set(available_fields)

    unavailable_fields = [
        field for field in requested_fields
        if field not in available_set
    ]

    if requested_fields:
        fields = [
            field for field in requested_fields
            if field in available_set
        ]
    else:
        fields = available_fields.copy()

    if "order_id" in available_set and "order_id" not in fields:
        fields.insert(0, "order_id")

    lines = []

    for item in orders:
        if not isinstance(item, dict) or not item.get("order_id"):
            continue

        details = [
            f"{_field_label(field)}: {_format_value(item.get(field))}"
            for field in fields
        ]

        if details:
            lines.append("- " + " · ".join(details))

    return lines, unavailable_fields


def _order_status_response(state):
    status = state.get("order_lookup_status")
    orders = state.get("order_info") or []
    reference = state.get("order_reference") or {}
    requested_fields = reference.get("requested_fields") or []

    if status == "NOT_FOUND":
        return (
            "I couldn't find that order for your signed-in account. Please check "
            "the order number or contact customer support.",
            "ORDER_NOT_FOUND",
            "Returned generic unavailable-order response",
            "NOT_GENERATED",
        )

    if status == "CLARIFY":
        options = [
            item for item in (state.get("clarification_options") or [])
            if isinstance(item, dict)
        ]

        if len(options) == 1:
            message = (
                "I found one possible order. Is this the one you mean?\n"
                f"{_order_option_label(options[0])}\n"
                "Reply ‘1’ to confirm, or provide another identifying detail."
            )
            summary = "Asked customer to confirm the closest matching order"
        elif options:
            message = (
                "I found a few possible orders. Which one do you mean?\n"
                + "\n".join(
                    f"{index}. {_order_option_label(item)}"
                    for index, item in enumerate(options, 1)
                )
                + "\n\nReply with the number or a detail such as ‘the older one’."
            )
            summary = "Asked customer to disambiguate among their orders"
        else:
            message = (
                "I couldn't identify which order you mean. Please provide "
                "an order number or another identifying detail."
            )
            summary = "Requested additional order identification"

        return message, "ORDER_CLARIFICATION", summary, "NOT_GENERATED"

    if status == "NO_MATCH":
        return (
            "I couldn't find any orders matching those criteria. "
            "Try broadening your search or changing one of the filters.",
            "ORDER_NO_MATCH",
            "No authenticated customer orders matched the requested criteria",
            "NOT_GENERATED",
        )

    if status == "INVALID_FILTER":
        return (
            "I couldn't apply one or more of those filters using the "
            "available order information. Please check the requested "
            "criteria or try a different filter.",
            "INVALID_ORDER_FILTER",
            "Order filter validation failed",
            "NOT_GENERATED",
        )

    if status in {"FOUND", "LISTED"}:
        if not isinstance(orders, list):
            orders = []

        orders = [
            item for item in orders
            if isinstance(item, dict) and item.get("order_id")
        ]

        lines, unavailable_fields = _order_lines(
            orders, requested_fields
        )

        if not lines:
            return (
                "I found the matching order, but couldn't retrieve its "
                "displayable details. Please try again.",
                "ORDER_DETAILS_UNAVAILABLE",
                "Order match contained no displayable details",
                "NOT_GENERATED",
            )

        message = (
            "Here are your matching orders:\n" + "\n".join(lines)
            if status == "LISTED"
            else "Here are the details of your order:\n" + "\n".join(lines)
        )

        if unavailable_fields:
            labels = ", ".join(
                _field_label(field) for field in unavailable_fields
            )
            message += (
                f"\n\nThe following requested details aren't available "
                f"in your order records: {labels}."
            )

        if status == "LISTED":
            workflow_status = "ORDERS_LISTED"
        elif len(lines) == 1:
            workflow_status = "ORDER_FOUND"
        else:
            message += "\n\nWhich order would you like help with?"
            workflow_status = "ORDERS_LISTED"

        return (
            message,
            workflow_status,
            "Displayed only authenticated customer's matching orders",
            "NOT_GENERATED",
        )

    fixed_responses = {
        "EMPTY": (
            "I couldn't find any orders for your signed-in account. "
            "Please contact customer support if you need help.",
            "NO_ORDERS",
            "No orders listed for authenticated customer",
        ),
        "SIGN_IN_REQUIRED": (
            SIGN_IN_REQUIRED_MESSAGE,
            "SIGN_IN_REQUIRED",
            "Sign-in required for order request",
        ),
    }

    response = fixed_responses.get(
        status,
        (
            SAFE_ERROR_RESPONSE,
            "ORDER_LOOKUP_ERROR",
            "Order lookup failed safely",
        ),
    )

    message, workflow_status, summary = response
    response_status = (
        "ERROR" if status not in fixed_responses else "NOT_GENERATED"
    )

    return message, workflow_status, summary, response_status
