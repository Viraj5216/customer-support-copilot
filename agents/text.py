
"""Text extraction, redaction, and conversation-history formatting."""

import json
import re


_CREDENTIAL_PATTERN = re.compile(
    r"\b(?:password|passwd|passcode|passphrase|otp|"
    r"one[- ]time\s+(?:password|code)|verification\s+code|"
    r"api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
    r"secret[_ -]?key|private[_ -]?key|client[_ -]?secret)\b"
    r"\s*(?:is|:|=)\s*([^\s,;]+)",
    re.I,
)

_BEARER_PATTERN = re.compile(
    r"\b(?:authorization\s*:\s*bearer|bearer)\s+"
    r"([A-Za-z0-9._~+/=-]+)",
    re.I,
)

_EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.I,
)

_PHONE_PATTERN = re.compile(
    r"(?<![\w-])(?:\+?\d[\d\s().-]{7,}\d)(?![\w-])"
)


def redact_user_secrets(text: str) -> str:
    """Mask credentials while preserving surrounding text."""
    text = str(text) if text is not None else ""

    text = _CREDENTIAL_PATTERN.sub(
        lambda match: (
            match.group(0)[:match.start(1) - match.start()]
            + "[credential omitted]"
        ),
        text,
    )

    return _BEARER_PATTERN.sub(
        lambda match: (
            match.group(0)[:match.start(1) - match.start()]
            + "[credential omitted]"
        ),
        text,
    )


def content_text(message) -> str:
    """Extract text from a message or supported content blocks."""
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                parts.append(item["text"])

        if parts:
            return "\n".join(parts)

    return json.dumps(content, ensure_ascii=False, default=str)


def model_safe_text(text: str) -> str:
    """Redact credentials, email addresses, and phone numbers."""
    text = redact_user_secrets(text)
    text = _EMAIL_PATTERN.sub("[email omitted]", text)
    return _PHONE_PATTERN.sub("[phone omitted]", text)


def format_history(history: list[dict]) -> str:
    """Format recent user and assistant messages for model prompts."""
    if not isinstance(history, (list, tuple)) or not history:
        return "No previous conversation."

    lines = []

    for message in history[-10:]:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).casefold().strip()
        if role not in {"user", "assistant"}:
            continue

        content = message.get("content", "")
        if not isinstance(content, str):
            content = content_text(content)

        content = model_safe_text(content.strip())[:4000]

        if content:
            lines.append(f"{role.upper()}: {content}")

    return "\n".join(lines) if lines else "No previous conversation."
