
from typing import Callable

from langgraph.graph import END, START, StateGraph

from agents.nodes import build_nodes
from agents.routing import (
    route_after_gatekeeper,
    route_after_order_lookup,
    route_after_response,
    route_after_retrieval,
    route_from_start,
)
from agents.schemas import CopilotState
from config import Settings, get_settings
from llm.gateway import LLMGateway
from llm.provider import get_chat_model


MAIN_NODES = (
    "gatekeeper_agent",
    "rag_agent",
    "order_lookup_agent",
    "response_agent",
)

RESPONSE_NODES = (
    "unsafe_response",
    "out_of_scope_response",
    "error_response",
    "no_policy_response",
    "order_status_response",
    "sign_in_required_response",
)

ROUTES = {
    START: (
        route_from_start,
        {
            "sign_in_required_response": "sign_in_required_response",
            "gatekeeper_agent": "gatekeeper_agent",
            "order_lookup_agent": "order_lookup_agent",
        },
    ),
    "gatekeeper_agent": (
        route_after_gatekeeper,
        {
            "rag_agent": "rag_agent",
            "order_lookup_agent": "order_lookup_agent",
            "response_agent": "response_agent",
            "sign_in_required_response": "sign_in_required_response",
            "unsafe_response": "unsafe_response",
            "out_of_scope_response": "out_of_scope_response",
            "error_response": "error_response",
        },
    ),
    "rag_agent": (
        route_after_retrieval,
        {
            "error_response": "error_response",
            "no_policy_response": "no_policy_response",
            "response_agent": "response_agent",
        },
    ),
    "order_lookup_agent": (
        route_after_order_lookup,
        {
            "error_response": "error_response",
            "order_status_response": "order_status_response",
            "rag_agent": "rag_agent",
            "response_agent": "response_agent",
        },
    ),
    "response_agent": (
        route_after_response,
        {
            "error_response": "error_response",
            "end": END,
        },
    ),
}


def build_graph(
    *,
    model=None,
    retriever=None,
    settings: Settings | None = None,
    llm_gateway: LLMGateway | None = None,
    clock: Callable | None = None,
):
    app_settings = settings or get_settings()
    model_name = getattr(model, "model", None) or app_settings.gemini_model

    if llm_gateway is None:

        def model_factory(name, timeout):
            if model is not None and name == model_name:
                return model
            return get_chat_model(name, timeout)

        llm_gateway = LLMGateway(app_settings, model_factory)

    nodes = build_nodes(
        app_settings=app_settings,
        model_name=model_name,
        model=model,
        gateway=llm_gateway,
        retriever=retriever,
        clock=clock,
    )

    workflow = StateGraph(CopilotState)

    for name in (*MAIN_NODES, *RESPONSE_NODES):
        workflow.add_node(name, nodes[name])

    for source, (router, destinations) in ROUTES.items():
        workflow.add_conditional_edges(
            source,
            router,
            destinations,
        )

    for name in RESPONSE_NODES:
        workflow.add_edge(name, END)

    return workflow.compile()
