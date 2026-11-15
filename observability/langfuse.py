"""Optional Langfuse tracing for LangGraph requests."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os

from config import Settings, get_settings

logger = logging.getLogger(__name__)


def create_langfuse_handler(settings: Settings | None = None):
    """Return one Langfuse LangChain callback, or None when tracing is unavailable."""
    config = settings or get_settings()
    if config.langfuse_public_key is None or config.langfuse_secret_key is None:
        return None

    try:
        # Import lazily: the app loads .env before the first customer request, and
        # the normal response path remains usable if the optional SDK is absent.
        os.environ.setdefault(
            "LANGFUSE_PUBLIC_KEY", config.langfuse_public_key.get_secret_value()
        )
        os.environ.setdefault(
            "LANGFUSE_SECRET_KEY", config.langfuse_secret_key.get_secret_value()
        )
        os.environ.setdefault("LANGFUSE_BASE_URL", config.langfuse_base_url)
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as exc:  # observability setup must never block support
        logger.warning("Langfuse callback unavailable (%s)", type(exc).__name__)
        return None


def pseudonymous_langfuse_user_id(user_id: str, salt: str | None) -> str | None:
    """Return a stable HMAC pseudonym; omit identity when no salt is configured."""
    if not user_id or not salt:
        return None
    return hmac.new(salt.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256).hexdigest()


def langfuse_stream_config(handler, session_id: str, user_id: str | None = None, salt: str | None = None) -> dict:
    """Build the single callback configuration for one customer turn."""
    metadata = {
        "langfuse_session_id": session_id,
        "langfuse_tags": ["customer-support-copilot"],
    }
    pseudonym = pseudonymous_langfuse_user_id(user_id or "", salt)
    if pseudonym:
        metadata["langfuse_user_id"] = pseudonym
    return {
        "callbacks": [handler],
        "run_name": "customer-support-request",
        "metadata": metadata,
    }


def flush_langfuse(handler) -> None:
    """Wait for this request's queued telemetry without affecting its response."""
    if handler is None:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:  # telemetry flush errors are deliberately non-fatal
        logger.warning("Langfuse flush failed (%s)", type(exc).__name__)
