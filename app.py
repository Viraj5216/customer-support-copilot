import logging
import time
import uuid

import streamlit as st
from dotenv import load_dotenv

from config import get_settings
from auth import (
    GENERIC_AUTH_ERROR,
    authenticate,
    clear_order_context,
    get_current_user_from_session,
    initialize_chat_state,
    logout,
    start_session,
)
from tools.orders import check_order
from observability.langfuse import (
    create_langfuse_handler,
    flush_langfuse,
    langfuse_stream_config,
)
from agents.graph import build_graph
from agents.text import redact_user_secrets
from agents.responses import SAFE_ERROR_RESPONSE
from tools.retriever import get_retriever_service

logger = logging.getLogger(__name__)

load_dotenv()

st.set_page_config(page_title="Customer Support Copilot", page_icon="🤖")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }
    [data-testid="stChatMessage"] {
        border-radius: 1rem;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.65rem;
    }
    [data-testid="stChatInput"] {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }
    [data-testid="stExpander"] {
        border-radius: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_graph():
    graph = build_graph()
    try:
        get_retriever_service()
    except Exception as exc:
        logger.warning("Retriever warmup failed (%s)", type(exc).__name__)
    return graph


def reset_conversation():
    st.session_state.conversation_history = []
    st.session_state.workflow_runs = []
    st.session_state.langfuse_session_id = str(uuid.uuid4())
    clear_order_context(st.session_state)
    st.rerun()


def render_run_details(run):
    result = run.get("result", {})
    events = result.get("execution", [])
    prompt_tokens = sum(event.get("prompt_tokens", 0) for event in events)
    completion_tokens = sum(event.get("completion_tokens", 0) for event in events)
    llm_calls = sum(event.get("llm_calls", 0) for event in events)

    with st.expander("⚙ Run details", expanded=False):
        st.caption(f"Authentication: {run.get('auth_state', 'anonymous')}")
        latency_col, calls_col, tokens_col = st.columns(3)
        latency_col.metric("Latency", f"{run.get('elapsed', 0):.1f}s")
        calls_col.metric("LLM calls", llm_calls)
        tokens_col.metric(
            "Tokens",
            f"{prompt_tokens:,} in · {completion_tokens:,} out",
        )

        if events:
            node_labels = {
                "gatekeeper_agent": "Gatekeeper",
                "rag_agent": "RAG",
                "order_lookup_agent": "Order lookup",
                "response_agent": "Response",
                "unsafe_response": "Safety response",
                "out_of_scope_response": "Scope response",
            }
            pipeline = "  →  ".join(
                f"{node_labels.get(event.get('node'), event.get('node', 'Step'))} "
                f"{event.get('latency_seconds', 0):.1f}s"
                + (" · error" if event.get("status") == "error" else "")
                for event in events
            )
            st.caption(pipeline)
            for event in events:
                for model_call in event.get("model_calls", []):
                    call_detail = (
                        f"Model call: {model_call.get('model') or 'unknown'} — "
                        f"{model_call.get('status')} — "
                        f"{model_call.get('duration_seconds', 0):.2f}s"
                    )
                    if model_call.get("http_status") is not None:
                        call_detail += f" — HTTP {model_call['http_status']}"
                    if model_call.get("provider_error_code"):
                        call_detail += f" — {model_call['provider_error_code']}"
                    if model_call.get("error_message"):
                        call_detail += f" — {model_call['error_message']}"
                    call_detail += f" — retries: {model_call.get('retry_attempts', 0)}"
                    if model_call.get("fallback_used"):
                        call_detail += f" — fallback: {model_call.get('final_model')}"
                    attempt_details = []
                    for attempt in model_call.get("attempts", []):
                        item = (
                            f"#{attempt.get('attempt')} {attempt.get('model')} "
                            f"{attempt.get('status')} "
                            f"{attempt.get('duration_seconds', 0):.2f}s"
                        )
                        if attempt.get("http_status") is not None:
                            item += f" HTTP {attempt['http_status']}"
                        if attempt.get("provider_error_code"):
                            item += f" {attempt['provider_error_code']}"
                        attempt_details.append(item)
                    if attempt_details:
                        call_detail += " — " + "; ".join(attempt_details)
                    if model_call.get("time_to_first_token_seconds") is not None:
                        call_detail += (
                            f" — first token: "
                            f"{model_call['time_to_first_token_seconds']:.2f}s"
                        )
                    st.caption(call_detail)

                if event.get("status") == "error":
                    error_type = event.get("error_type")
                    detail = f" · {error_type}" if error_type else ""
                    st.caption(
                        f"{event.get('node')} — error — "
                        f"{event.get('latency_seconds', 0):.2f}s — "
                        f"{event.get('summary', 'Step failed')}{detail}"
                    )
                    provider_detail = " · ".join(
                        f"{label}: {event[key]}"
                        for key, label in (
                            ("error_status", "HTTP status"),
                            ("provider_error_code", "Provider code"),
                            ("error_message", "Provider message"),
                        )
                        if event.get(key) is not None
                    )
                    if provider_detail:
                        st.caption(provider_detail)

        response_event = next(
            (
                event
                for event in reversed(events)
                if event.get("node") == "response_agent"
            ),
            None,
        )
        for label, key in (
            ("Retrieval", "retrieval_status"),
            ("Order lookup", "order_lookup_status"),
            ("Response generation", "response_status"),
        ):
            status = result.get(key)
            if status:
                if key == "order_lookup_status":
                    status = {
                        "SIGN_IN_REQUIRED": "sign-in required",
                        "FOUND": "found for signed-in customer",
                        "NOT_FOUND": "not found for signed-in customer",
                        "LISTED": "customer orders listed",
                        "EMPTY": "no customer orders",
                    }.get(status, status.replace("_", " ").lower())
                else:
                    status = status.replace("_", " ").lower()
                st.caption(f"{label}: {status}")

        errors = result.get("errors", [])
        if errors:
            st.caption("One or more workflow steps reported an issue.")

    evidence = result.get("evidence", [])
    if evidence:
        with st.expander("🔍 Sources", expanded=False):
            for item in evidence:
                title = " · ".join(
                    part
                    for part in (item.get("section"), item.get("subsection"))
                    if part
                )
                st.markdown(f"**{title or 'Policy source'}**")
                if item.get("distance") is not None:
                    distance = item["distance"]
                    max_distance = settings.retrieval_max_distance or 1.35
                    confidence = (
                        "High"
                        if distance <= settings.retrieval_high_confidence_distance
                        else "Medium"
                        if distance <= max_distance
                        else "Low"
                    )
                    st.caption(
                        f"{confidence} confidence · relevance distance: {distance:.4f}"
                    )
                if item.get("rerank_score") is not None:
                    st.caption(f"Reranker score: {item['rerank_score']:.4f}")
                if item.get("text"):
                    st.write(item["text"])


def render_order_choices(run, run_index):
    result = run.get("result", {})
    pending = st.session_state.get("pending_order_clarification_context")
    if not pending or pending.get("original_question") != run.get("question"):
        return
    for option in result.get("clarification_options", []):
        if st.button(
            option["label"], key=f"order-option-{run_index}-{option['order_id']}"
        ):
            signed_in = get_current_user_from_session(st.session_state)
            validated = (
                check_order(option["order_id"], signed_in["user_id"], settings)
                if signed_in
                else {"found": False}
            )
            if validated.get("found"):
                st.session_state.current_order = validated["order_id"]
                st.session_state.pending_order_clarification = {
                    "original_question": pending["original_question"],
                    "options": pending["options"],
                    "selected_order_id": validated["order_id"],
                }
                st.rerun()


settings = get_settings()
if settings.gemini_api_key is None:
    st.error("GEMINI_API_KEY is not set. Add it to your .env file.")
    st.stop()

initialize_chat_state(st.session_state)
for saved_message in st.session_state.conversation_history:
    if isinstance(saved_message.get("content"), str):
        saved_message["content"] = redact_user_secrets(saved_message["content"])

with st.sidebar:
    st.markdown("### 💬 Support chat")
    current_user = get_current_user_from_session(st.session_state)
    if current_user:
        st.caption("Signed in")
        if st.button("Sign out", use_container_width=True):
            logout(st.session_state)
            st.rerun()
    else:
        st.caption(
            "Sign in to view your orders. Policy and general help remain available anonymously."
        )
        with st.form("customer_login", clear_on_submit=True):
            login_email = st.text_input("Email")
            login_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            try:
                user = authenticate(login_email, login_password, settings)
                start_session(st.session_state, user)
                st.rerun()
            except ValueError:
                st.error(GENERIC_AUTH_ERROR)
    st.button(
        "＋  New chat",
        type="primary",
        use_container_width=True,
        on_click=reset_conversation,
    )
    st.button(
        "Delete conversation",
        use_container_width=True,
        on_click=reset_conversation,
    )

# The authentication check above can expire an idle session and call logout(),
# which clears these keys. Recreate an empty anonymous chat before rendering.
initialize_chat_state(st.session_state)

st.title("Customer Support")
st.caption("Ask about an order, a return, or a product issue.")

# Render the persistent transcript, with each turn's diagnostics kept tucked
# beneath its assistant response.
legacy_run_index = 0
for message in st.session_state.conversation_history:
    role = message.get("role", "assistant")
    with st.chat_message(role, avatar="🤖" if role == "assistant" else None):
        st.markdown(message.get("content", ""))
    if role == "assistant":
        run_index = message.get("run_index")
        if not isinstance(run_index, int):
            run_index = legacy_run_index
        if 0 <= run_index < len(st.session_state.workflow_runs):
            saved_run = st.session_state.workflow_runs[run_index]
            render_run_details(saved_run)
            render_order_choices(saved_run, run_index)
        legacy_run_index += 1

question = st.chat_input("Message Customer Support…")

resume = st.session_state.pop("pending_order_clarification", None)
typed_pending = st.session_state.get("pending_order_clarification_context")
active_question = resume["original_question"] if resume else question
active_selection = resume.get("selected_order_id") if resume else None
pending_context = resume or typed_pending
pending_original_question = (
    pending_context.get("original_question") if pending_context else None
)

if active_question and active_question.strip():
    user_message = redact_user_secrets(active_question.strip())
    submitted_message = (
        redact_user_secrets(question.strip())
        if question and not resume
        else user_message
    )
    if not resume and typed_pending:
        user_message = submitted_message
    current_user = get_current_user_from_session(st.session_state)
    user_id = current_user["user_id"] if current_user else None
    auth_state = "signed in" if current_user else "anonymous"
    prior_history = [
        {
            "role": item.get("role", ""),
            "content": item.get("content", ""),
        }
        for item in st.session_state.conversation_history
    ]
    if not resume:
        st.session_state.conversation_history.append(
            {"role": "user", "content": submitted_message}
        )

    with st.chat_message("user"):
        st.markdown(
            submitted_message
            if not resume
            else f"Selected an order for: {user_message}"
        )

    graph = get_graph()
    state = {
        "user_message": user_message,
        "user_id": user_id,
        "auth_state": auth_state,
        "conversation_history": prior_history,
        "execution": [],
        "errors": [],
        "current_order": st.session_state.current_order,
    }
    if pending_context:
        state.update(
            {
                "clarification_options": pending_context["options"],
                "clarification_original_question": redact_user_secrets(
                    pending_original_question
                ),
                "selected_order_id": active_selection,
            }
        )
        st.session_state.pop("pending_order_clarification_context", None)
    started = time.perf_counter()
    result = None
    streamed_response = ""
    langfuse_handler = create_langfuse_handler(settings)
    run_index = len(st.session_state.workflow_runs)

    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        try:
            with st.spinner("Thinking…"):
                stream_kwargs = {"stream_mode": ["values", "custom"]}
                trace_config = (
                    langfuse_stream_config(
                        langfuse_handler,
                        st.session_state.langfuse_session_id,
                        user_id=user_id,
                        salt=(
                            settings.langfuse_user_id_salt.get_secret_value()
                            if settings.langfuse_user_id_salt
                            else None
                        ),
                    )
                    if langfuse_handler is not None
                    else {}
                )
                # The authenticated user ID is graph state from this server-side session.
                # It is never included in model messages or prompt context.
                stream_kwargs["config"] = trace_config
                stream = graph.stream(state, **stream_kwargs)
                for mode, chunk in stream:
                    if mode == "custom":
                        if (
                            isinstance(chunk, dict)
                            and chunk.get("type") == "response_token"
                        ):
                            streamed_response += chunk.get("text", "")
                            response_placeholder.markdown(
                                streamed_response.replace("TERMINATE", "")
                            )
                        continue

                    if isinstance(chunk, dict):
                        result = chunk

            elapsed = time.perf_counter() - started
            if result is None:
                raise RuntimeError("Workflow returned no result")

            final_response = result.get("response", "")
            if not final_response:
                final_response = streamed_response.replace("TERMINATE", "").strip()
            if not final_response:
                final_response = (
                    "I’m sorry, but I could not prepare a response. Please try again."
                )

            response_placeholder.markdown(final_response)
            st.session_state.conversation_history.append(
                {
                    "role": "assistant",
                    "content": final_response,
                    "run_index": run_index,
                }
            )
            if result.get("order_lookup_status") in (
                "FOUND",
                "NOT_FOUND",
                "NO_MATCH",
                "CLARIFY",
            ):
                st.session_state.current_order = result.get("current_order")
            run = {
                "result": result,
                "elapsed": elapsed,
                "auth_state": auth_state,
                "question": pending_original_question or user_message,
            }
            st.session_state.workflow_runs.append(run)
            if result.get("order_lookup_status") == "CLARIFY":
                st.session_state.pending_order_clarification_context = {
                    "original_question": pending_original_question or user_message,
                    "options": result.get("clarification_options", []),
                }
            render_run_details(run)
            render_order_choices(run, run_index)

        except Exception as exc:
            elapsed = time.perf_counter() - started
            failure_message = SAFE_ERROR_RESPONSE
            response_placeholder.markdown(failure_message)
            st.session_state.conversation_history.append(
                {
                    "role": "assistant",
                    "content": failure_message,
                    "run_index": run_index,
                }
            )
            run = {
                "result": {
                    "execution": [],
                    "errors": [type(exc).__name__],
                },
                "elapsed": elapsed,
                "auth_state": auth_state,
            }
            st.session_state.workflow_runs.append(run)
            render_run_details(run)
        finally:
            flush_langfuse(langfuse_handler)
