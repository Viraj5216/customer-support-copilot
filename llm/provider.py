
"""Google Gemini model factory."""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from config import get_settings


@lru_cache(maxsize=16)
def get_chat_model(
    model_name: str | None = None,
    timeout_seconds: float | None = None,
) -> ChatGoogleGenerativeAI:
    """Build a cached Gemini model; retries are managed by the gateway."""
    settings = get_settings()

    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    return ChatGoogleGenerativeAI(
        model=model_name or settings.gemini_model,
        google_api_key=settings.gemini_api_key.get_secret_value(),
        temperature=0,
        max_retries=0,
        timeout=(
            timeout_seconds
            if timeout_seconds is not None
            else settings.llm_timeout_seconds
        ),
    )
