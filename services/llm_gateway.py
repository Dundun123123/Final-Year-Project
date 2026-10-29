"""LangChain adapter for the selected DeepSeek chat model."""

from __future__ import annotations

import os
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek


logger = logging.getLogger(__name__)


def make_deepseek_model(model_name: str, max_tokens: int = 800) -> ChatDeepSeek | None:
    """Return the configured model, or None when no API key is configured."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return ChatDeepSeek(
        model=model_name,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=45,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
    )


def _message(role: str, content: Any) -> SystemMessage | HumanMessage | AIMessage:
    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    raise ValueError(f"Unsupported chat role: {role}")


def call_deepseek(messages: list[dict[str, Any]], model_name: str, max_tokens: int = 800) -> str | None:
    """Keep the existing prompt-call interface for Markdown and translation flows."""
    try:
        model = make_deepseek_model(model_name, max_tokens)
        if model is None:
            return None
        response = model.invoke([_message(item["role"], item["content"]) for item in messages])
        if isinstance(response.content, str):
            return response.content.strip() or None
        text = "".join(part.get("text", "") for part in response.content if isinstance(part, dict))
        return text.strip() or None
    except Exception as exc:
        # Preserve the existing local fallback when the API is unavailable.
        logger.warning("DeepSeek call unavailable (%s); using local fallback", type(exc).__name__)
        return None
