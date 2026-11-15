"""Demo-grade local authentication for Streamlit sessions."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

from config import Settings, get_settings

GENERIC_AUTH_ERROR = "Email or password is incorrect."
_CHAT_KEYS = (
    "conversation_history", "workflow_runs", "langfuse_session_id",
    "authenticated_user_id", "authenticated_display_name", "auth_last_activity",
    "current_order", "pending_order_clarification", "pending_order_clarification_context",
)


@dataclass
class _FailureState:
    failures: int = 0
    locked_until: float = 0.0


class AuthService:
    """Local Argon2id authentication with process-local account lockouts."""

    def __init__(
        self,
        database_path: Path,
        *,
        max_failures: int = 5,
        lock_seconds: int = 300,
        idle_timeout_seconds: int = 1800,
        now=time.time,
        hasher: PasswordHasher | None = None,
    ):
        self.database_path = Path(database_path)
        self.max_failures = max_failures
        self.lock_seconds = lock_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        self.now = now
        self.hasher = hasher or PasswordHasher()
        self._lock = threading.Lock()
        self._failures: dict[str, _FailureState] = {}
        self._dummy_hash = self.hasher.hash("unusable-dummy-password")

    def authenticate(self, email: str, password: str) -> dict:
        normalized = (email or "").strip().casefold()
        # Hash the normalized account identifier so lockout storage has no email.
        import hashlib
        account_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        now = self.now()
        with self._lock:
            attempt = self._failures.setdefault(account_key, _FailureState())
            if attempt.locked_until > now:
                raise ValueError(GENERIC_AUTH_ERROR)
            if attempt.locked_until:
                attempt.failures = 0
                attempt.locked_until = 0

        record = self._find_user(normalized)
        encoded_hash = record[2] if record else self._dummy_hash
        valid = False
        try:
            valid = self.hasher.verify(encoded_hash, password or "")
        except (VerificationError, VerifyMismatchError, ValueError):
            valid = False
        # Argon2id's verify routine performs the password-hash comparison; unknown
        # accounts were checked against the same-cost dummy hash above.
        if not record or not valid:
            with self._lock:
                attempt = self._failures.setdefault(account_key, _FailureState())
                attempt.failures += 1
                if attempt.failures >= self.max_failures:
                    attempt.locked_until = self.now() + self.lock_seconds
            raise ValueError(GENERIC_AUTH_ERROR)
        with self._lock:
            self._failures.pop(account_key, None)
        return {"user_id": record[0], "display_name": record[1]}

    def _find_user(self, normalized_email: str):
        if not self.database_path.is_file():
            return None
        import sqlite3
        uri = f"file:{self.database_path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            return connection.execute(
                "SELECT user_id, display_name, password_hash FROM users "
                "WHERE login_email = ?", (normalized_email,),
            ).fetchone()

    def get_current_user(self, session_state, *, now: float | None = None):
        user_id = session_state.get("authenticated_user_id")
        if not user_id:
            return None
        current = self.now() if now is None else now
        last = session_state.get("auth_last_activity", current)
        if current - float(last) >= self.idle_timeout_seconds:
            logout(session_state)
            return None
        session_state["auth_last_activity"] = current
        return {
            "user_id": user_id,
            "display_name": session_state.get("authenticated_display_name", "Customer"),
        }


@lru_cache(maxsize=4)
def get_auth_service(database_path: str, idle_timeout_seconds: int = 1800) -> AuthService:
    return AuthService(Path(database_path), idle_timeout_seconds=idle_timeout_seconds)


def authenticate(email: str, password: str, settings: Settings | None = None) -> dict:
    config = settings or get_settings()
    service = get_auth_service(
        str(config.orders_db_path), config.auth_session_idle_timeout_seconds
    )
    return service.authenticate(email, password)


def get_current_user(session_state, *, now: float | None = None, idle_timeout_seconds=1800):
    user_id = session_state.get("authenticated_user_id")
    if not user_id:
        return None
    current = time.time() if now is None else now
    last = session_state.get("auth_last_activity", current)
    if current - float(last) >= idle_timeout_seconds:
        logout(session_state)
        return None
    session_state["auth_last_activity"] = current
    return {
        "user_id": user_id,
        "display_name": session_state.get("authenticated_display_name", "Customer"),
    }


def logout(session_state) -> None:
    """Clear identity and all per-user chat/trace state."""
    for key in _CHAT_KEYS:
        session_state.pop(key, None)


def clear_order_context(session_state) -> None:
    """Clear resolved and pending order references for New chat/Delete conversation."""
    session_state["current_order"] = None
    session_state.pop("pending_order_clarification", None)
    session_state.pop("pending_order_clarification_context", None)


def initialize_chat_state(session_state) -> None:
    """Ensure chat UI state exists after auth checks may have logged out."""
    session_state.setdefault("conversation_history", [])
    session_state.setdefault("workflow_runs", [])
    session_state.setdefault("langfuse_session_id", str(uuid.uuid4()))
    session_state.setdefault("current_order", None)


def start_session(session_state, user: dict, *, now: float | None = None) -> None:
    # A fresh principal never inherits the previous principal's transcript.
    logout(session_state)
    session_state["authenticated_user_id"] = user["user_id"]
    session_state["authenticated_display_name"] = user["display_name"]
    session_state["auth_last_activity"] = time.time() if now is None else now
    session_state["conversation_history"] = []
    session_state["workflow_runs"] = []
    session_state["langfuse_session_id"] = str(uuid.uuid4())


# The Streamlit adapter calls this interface; it can later be replaced by OIDC.
def get_current_user_from_session(session_state):
    timeout = get_settings().auth_session_idle_timeout_seconds
    return get_current_user(session_state, idle_timeout_seconds=timeout)
