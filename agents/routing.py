
"""Named conditional routes for the support workflow."""

import re

from agents.schemas import CopilotState


_ORDER_REFERENCE_FIELDS = (
    "order_id",
    "product",
    "variant",
    "status",
    "date_field",
    "date_from",
    "date_to",
    "position_index",
    "sort_by",
    "sort_order",
    "filters",
    "requested_fields",
    "wants_all",
    "list_orders",
    "ambiguous",
    "needs_clarification",
    "clarification_choice",
)


_ORDER_LOOKUP_STATUSES = {
    "NOT_FOUND",
    "LISTED",
    "EMPTY",
    "SIGN_IN_REQUIRED",
    "CLARIFY",
    "NO_MATCH",
    "FOUND",
    "INVALID_FILTER",
}


def _has_safe_information_question(state: CopilotState) -> bool:
    """Allow an independent information question through an overbroad refusal."""
    message = str(state.get("user_message", ""))
    if not re.search(
        r"\?|\b(?:how do i|how can i|what is|what are|why is|why are|"
        r"when is|where can i|explain|tell me about)\b",
        message,
        re.I,
    ):
        return False

    security_request = re.search(
        r"\b(?:ignore (?:all )?(?:previous|prior|above)|pretend you are|"
        r"impersonate|bypass|hack|steal|all customers|every customer|"
        r"another customer|other customer|another user|other user|"
        r"someone else|someone else's|another person's|other people's|"
        r"my friend|my spouse|their (?:order|account|address|billing|payment|"
        r"password|email|phone|private information|data)|"
        r"system prompt|hidden instructions)\b",
        message,
        re.I,
    )
    return security_request is None


def route_from_start(state: CopilotState) -> str:
    return "gatekeeper_agent"


def route_after_gatekeeper(state: CopilotState) -> str:
    if (
        state.get("gatekeeper_status") == "ERROR"
        or state.get("safety") == "ERROR"
    ):
        return "error_response"

    if state.get("safety") != "SAFE":
        if _has_safe_information_question(state):
            return "rag_agent"
        return "unsafe_response"

    scope = state.get("scope")

    if scope == "order_status_lookup":
        return (
            "order_lookup_agent"
            if state.get("user_id")
            else "sign_in_required_response"
        )

    if scope == "in_scope":
        reference = state.get("order_reference") or {}
        has_order_reference = any(
            reference.get(field)
            for field in _ORDER_REFERENCE_FIELDS
        )

        if has_order_reference or state.get("references_current_order"):
            return (
                "order_lookup_agent"
                if state.get("user_id")
                else "sign_in_required_response"
            )

        if state.get("requires_policy") is True:
            return "rag_agent"

        return "response_agent"

    if scope == "out_of_scope":
        return "out_of_scope_response"

    return "error_response"


def route_after_order_lookup(state: CopilotState) -> str:
    status = state.get("order_lookup_status")

    if status not in _ORDER_LOOKUP_STATUSES:
        return "error_response"

    if status not in {"FOUND", "LISTED"}:
        return "order_status_response"

    if state.get("requires_policy") is True:
        return "rag_agent"

    return "response_agent"


def route_after_retrieval(state: CopilotState) -> str:
    status = state.get("retrieval_status")

    if status == "SUCCESS":
        return "response_agent"

    if status in {"NO_RELEVANT_POLICY_FOUND", "ERROR"}:
        if state.get("order_lookup_status") in {"FOUND", "LISTED"}:
            return "response_agent"

        if status == "NO_RELEVANT_POLICY_FOUND":
            return "no_policy_response"

    return "error_response"


def route_after_response(state: CopilotState) -> str:
    if state.get("response_status") == "SUCCESS":
        return "end"

    return "error_response"
