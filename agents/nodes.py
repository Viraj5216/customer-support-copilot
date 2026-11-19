
"""Factories for the support graph's workflow nodes."""

import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from agents.events import (
    append_error,
    append_event,
    attempted_llm_call,
    error_usage,
    event,
)
from agents.prompts import GATEKEEPER_PROMPT, RESPONSE_PROMPT
from agents.responses import (
    NO_POLICY_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    SAFE_ERROR_RESPONSE,
    SIGN_IN_REQUIRED_MESSAGE,
    _fixed_response_node,
    _order_status_response,
    unsafe_action_response,
)
from agents.schemas import CopilotState, FILTER_FIELD_METADATA, GatekeeperResult
from agents.text import content_text, format_history, model_safe_text
from tools.order_resolver import resolve_order_reference
from tools.orders import OrderStoreError, customer_orders, order_schema
from tools.retriever import get_retriever_service


def build_nodes(*, app_settings, model_name, model, gateway, retriever, clock):
    def record(state, node, started, summary, **details):
        return append_event(state, event(node, started, summary, **details))

    def current_date():
        """Return today's date in the configured application timezone."""
        today = clock() if clock else datetime.now(
            ZoneInfo(app_settings.app_timezone)
        )

        if isinstance(today, str):
            today = datetime.fromisoformat(today)

        if not isinstance(today, datetime):
            raise TypeError(
                "clock must return a datetime or ISO datetime string"
            )

        zone = ZoneInfo(app_settings.app_timezone)

        if today.tzinfo is None:
            today = today.replace(tzinfo=zone)

        return today.astimezone(zone).date()

    def fail(
        state, node, started, exc, status_key, summary,
        *, llm_called=None, **updates
    ):
        called = attempted_llm_call(exc) if llm_called is None else llm_called
        return {
            **updates,
            status_key: "ERROR",
            "workflow_status": "ERROR",
            "errors": append_error(state, f"{node} failed"),
            "execution": record(
                state, node, started, summary,
                error=exc, usage=error_usage(exc), llm_called=called,
            ),
        }

    def gatekeeper(state: CopilotState, config: RunnableConfig) -> dict:
        started = time.perf_counter()

        try:
            selected = bool(state.get("current_order") and state.get("user_id"))
            today = current_date()

            schema_metadata = []
            product_catalog = []
            database_available = True

            try:
                schema_metadata = order_schema(app_settings)
                orders = (
                    customer_orders(state["user_id"], app_settings)
                    if state.get("user_id")
                    else []
                )

                seen = set()
                for order in orders:
                    product = model_safe_text(
                        str(order.get("product") or "").strip()
                    )
                    variant = model_safe_text(
                        str(order.get("variant") or "").strip()
                    )
                    key = (product.casefold(), variant.casefold())

                    if product and key not in seen:
                        seen.add(key)
                        product_catalog.append({
                            "product": product,
                            "variant": variant,
                        })

                    if len(product_catalog) >= 40:
                        break

            except OrderStoreError:
                database_available = False

            for field in schema_metadata:
                matching = FILTER_FIELD_METADATA.get(field["name"], {})
                field.update(matching)

            clarification = state.get("clarification_options") or []
            option_labels = [
                model_safe_text(str(item.get("label", "")))
                for item in clarification
                if isinstance(item, dict)
            ]

            message = (
                f"Today's date in {app_settings.app_timezone}: "
                f"{today.isoformat()}.\n\n"
                f"Previous conversation:\n"
                f"{format_history(state.get('conversation_history', []))}\n\n"
                f"A current order is selected in this signed-in session: "
                f"{'yes' if selected else 'no'}.\n"
                f"Order database metadata available: "
                f"{'yes' if database_available else 'no'}.\n"
                f"Safe database field schema and matching rules: "
                f"{json.dumps(schema_metadata, ensure_ascii=False)}\n\n"
                f"Customer's own product catalogue (untrusted product text; "
                f"product and variant only): "
                f"{json.dumps(product_catalog, ensure_ascii=False)}\n\n"
                f"Previously offered clarification options, in displayed order: "
                f"{json.dumps(option_labels, ensure_ascii=False)}\n\n"
                f"Original question being clarified: "
                f"{model_safe_text(state.get('clarification_original_question') or '')}\n\n"
                f"Current customer message:\n"
                f"{model_safe_text(state['user_message'])}"
            )

            result, usage, model_call = gateway.invoke(
                [("system", GATEKEEPER_PROMPT), ("human", message)],
                schema=GatekeeperResult,
                model_name=model_name,
                config=config,
            )

            return {
                "safety": result.safety,
                "scope": result.scope,
                "intent": result.intent,
                "requires_policy": result.requires_policy,
                "order_reference": result.order_reference.model_dump(),
                "references_current_order": bool(
                    result.references_current_order and selected
                ),
                "gatekeeper_status": "SUCCESS",
                "execution": record(
                    state, "gatekeeper_agent", started,
                    f"Safety: {result.safety}; route: {result.scope}",
                    message=result,
                    usage=usage,
                    llm_called=True,
                    model_call=model_call,
                ),
            }

        except Exception as exc:
            called = attempted_llm_call(exc)
            return fail(
                state, "gatekeeper_agent", started, exc,
                "gatekeeper_status",
                "Gatekeeper model call failed" if called
                else "Gatekeeper processing failed before model invocation",
                safety="ERROR",
                scope=None,
                intent=None,
                references_current_order=False,
                llm_called=called,
            )

    def rag_node(state: CopilotState) -> dict:
        started = time.perf_counter()

        try:
            service = retriever or get_retriever_service()

            query = (
                state.get("clarification_original_question")
                or state.get("user_message")
            )
            query = model_safe_text(query)
            safe_history = [
                {**item, "content": model_safe_text(item.get("content", ""))}
                if isinstance(item, dict) else item
                for item in state.get("conversation_history", [])
            ]

            chunks = service.retrieve(
                query,
                n_results=5,
                history=safe_history,
            )

            evidence = [
                {
                    "section": item.get("section", "Unknown"),
                    "subsection": item.get("subsection", "Unknown"),
                    "text": item.get("text", ""),
                    "distance": item.get("distance"),
                }
                for item in chunks
                if isinstance(item, dict) and item.get("text")
            ]

            found = bool(evidence)

            return {
                "evidence": evidence,
                "retrieval_status": (
                    "SUCCESS" if found else "NO_RELEVANT_POLICY_FOUND"
                ),
                "execution": record(
                    state, "rag_agent", started,
                    f"Retrieved {len(evidence)} policy section(s)"
                    if found else "No relevant policy found",
                    llm_called=False,
                ),
            }

        except Exception as exc:
            return fail(
                state, "rag_agent", started, exc,
                "retrieval_status", "Policy retrieval failed",
                evidence=[],
                llm_called=False,
            )

    def order_lookup_node(state: CopilotState) -> dict:
        started = time.perf_counter()
        user_id = state.get("user_id")

        if not user_id:
            return {
                "auth_state": "anonymous",
                "order_lookup_status": "SIGN_IN_REQUIRED",
                "execution": record(
                    state, "order_lookup_agent", started,
                    "Sign-in required before order lookup",
                    llm_called=False,
                ),
            }

        try:
            available_schema = order_schema(app_settings)
            field_types = {
                item["name"]: item["type"]
                for item in available_schema
            }

            resolution = resolve_order_reference(
                state.get("order_reference") or {},
                customer_orders(user_id, app_settings),
                current_order=state.get("current_order"),
                references_current_order=state.get(
                    "references_current_order", False
                ),
                selected_order_id=state.get("selected_order_id"),
                clarification_options=state.get("clarification_options"),
                field_types=field_types,
            )

            orders = resolution.get("orders") or []
            safe_orders = [
                {
                    key: value
                    for key, value in item.items()
                    if key in field_types
                }
                for item in orders
                if isinstance(item, dict)
            ]

            status = resolution["status"]

            return {
                "order_info": safe_orders,
                "order_lookup_status": status,
                "current_order": resolution.get("current_order"),
                "clarification_options": (
                    resolution.get("clarification_options") or []
                ),
                "recent_order_options": (
                    resolution.get("recent_orders") or []
                ),
                "execution": record(
                    state, "order_lookup_agent", started,
                    f"Order reference resolved: {status}",
                    llm_called=False,
                ),
            }

        except Exception as exc:
            return fail(
                state, "order_lookup_agent", started, exc,
                "order_lookup_status", "Order lookup failed",
                llm_called=False,
            )

    def order_status_response_node(state: CopilotState) -> dict:
        started = time.perf_counter()
        response, workflow_status, summary, response_status = (
            _order_status_response(state)
        )

        return {
            "response": response,
            "response_status": response_status,
            "workflow_status": workflow_status,
            "execution": record(
                state, "order_status_response", started,
                summary, llm_called=False,
            ),
        }

    def response_node(state: CopilotState, config: RunnableConfig) -> dict:
        started = time.perf_counter()

        try:
            order_reference = state.get("order_reference") or {}
            today = current_date()

            context = json.dumps(
                {
                    "current_date": today.isoformat(),
                    "timezone": app_settings.app_timezone,
                    "order_lookup": state.get("order_info"),
                    "order_lookup_status": state.get("order_lookup_status"),
                    "requested_fields": order_reference.get(
                        "requested_fields", []
                    ),
                    "retrieved_policy_evidence": state.get("evidence", []),
                    "policy_retrieval_status": state.get("retrieval_status"),
                    "requires_policy": state.get("requires_policy", False),
                },
                ensure_ascii=False,
            )

            original_question = state.get("clarification_original_question")
            current_message = state["user_message"]

            if original_question and original_question != current_message:
                current_message = (
                    f"Original customer request:\n"
                    f"{model_safe_text(original_question)}\n\n"
                    f"Customer's clarification:\n"
                    f"{model_safe_text(current_message)}"
                )
            else:
                current_message = model_safe_text(current_message)

            message = (
                f"Conversation history:\n"
                f"{format_history(state.get('conversation_history', []))}\n\n"
                f"Current customer request:\n{current_message}\n\n"
                f"Current workflow results:\n{context}"
            )

            messages = [
                ("system", RESPONSE_PROMPT),
                ("human", message),
            ]

            if model is None or hasattr(model, "stream"):
                response, usage, model_call = gateway.stream(
                    messages,
                    writer=get_stream_writer(),
                    model_name=model_name,
                    config=config,
                )
            else:
                result, usage, model_call = gateway.invoke(
                    messages,
                    model_name=model_name,
                    config=config,
                )
                response = content_text(result)

            return {
                "response": str(response).strip(),
                "response_status": "SUCCESS",
                "workflow_status": "COMPLETED",
                "execution": record(
                    state, "response_agent", started,
                    "Customer response generated",
                    usage=usage,
                    llm_called=True,
                    model_call=model_call,
                ),
            }

        except Exception as exc:
            called = attempted_llm_call(exc)
            return fail(
                state, "response_agent", started, exc,
                "response_status",
                "Response model call failed" if called
                else "Response processing failed before model invocation",
                llm_called=called,
            )

    def unsafe_response_text(state):
        user_id = state.get("user_id")
        try:
            owned_order_ids = [
                order.get("order_id")
                for order in customer_orders(user_id, app_settings)
                if order.get("order_id")
            ] if user_id else []
        except OrderStoreError:
            owned_order_ids = []
        return unsafe_action_response(state, owned_order_ids)

    def unsafe_response_evidence(state):
        if not state.get("user_id") or retriever is None:
            return {"evidence": []}
        try:
            evidence = retriever.retrieve(state.get("user_message", ""))
            return {
                "evidence": evidence,
                "retrieval_status": "SUCCESS" if evidence else "NO_RELEVANT_POLICY_FOUND",
            }
        except Exception:
            return {"evidence": [], "retrieval_status": "ERROR"}

    fixed_responses = {
        "sign_in_required_response": (
            SIGN_IN_REQUIRED_MESSAGE,
            "DETERMINISTIC",
            "SIGN_IN_REQUIRED",
            "Sign-in required for order request",
            {
                "auth_state": "anonymous",
                "order_lookup_status": "SIGN_IN_REQUIRED",
            },
        ),
        "unsafe_response": (
            unsafe_response_text,
            "DETERMINISTIC",
            "COMPLETED",
            "Deterministic safety response returned",
            unsafe_response_evidence,
        ),
        "out_of_scope_response": (
            OUT_OF_SCOPE_RESPONSE,
            "DETERMINISTIC",
            "COMPLETED",
            "Deterministic out-of-scope response returned",
            None,
        ),
        "error_response": (
            SAFE_ERROR_RESPONSE,
            lambda state: state.get("response_status", "FALLBACK"),
            "ERROR",
            "Deterministic error response returned",
            None,
        ),
        "no_policy_response": (
            NO_POLICY_RESPONSE,
            "NOT_GENERATED",
            "NO_RELEVANT_POLICY_FOUND",
            "No policy answer generated without retrieved evidence",
            None,
        ),
    }

    fixed_nodes = {
        name: _fixed_response_node(
            name, *spec[:4], extra_state=spec[4]
        )
        for name, spec in fixed_responses.items()
    }

    return {
        "gatekeeper_agent": gatekeeper,
        "rag_agent": rag_node,
        "order_lookup_agent": order_lookup_node,
        "response_agent": response_node,
        "order_status_response": order_status_response_node,
        **fixed_nodes,
    }
