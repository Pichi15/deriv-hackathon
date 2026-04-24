"""Thin wrapper over ChatOpenAI pointed at the LiteLLM proxy.

All three LLM-calling steps (term id, translation, QA) route through the same
OpenAI-compatible endpoint. Keeping one factory avoids duplicated client config
and centralises the token-usage plumbing used by cost logging.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from PIL import Image

from .config import LITELLM_API_KEY, LITELLM_BASE_URL

# Anthropic caps images at 8000px on any side.
MAX_IMAGE_DIM = 7500


def make_chat(model: str, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
    )


def encode_image(path: str | Path) -> str:
    """Return a data URL for a PNG screenshot — used as a vision content block.

    Full-page Playwright screenshots can be taller than the 8000px Anthropic
    vision limit. Downscale proportionally if either dimension exceeds the cap.
    """
    img = Image.open(Path(path))
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def extract_usage(response: Any) -> tuple[int, int]:
    """Pull (input_tokens, output_tokens) from a LangChain response.

    ChatOpenAI surfaces usage two ways depending on version — prefer
    `usage_metadata`, fall back to `response_metadata.token_usage`.
    """
    usage_meta = getattr(response, "usage_metadata", None) or {}
    if usage_meta:
        return (
            int(usage_meta.get("input_tokens", 0) or 0),
            int(usage_meta.get("output_tokens", 0) or 0),
        )
    rm = getattr(response, "response_metadata", {}) or {}
    tu = rm.get("token_usage", {}) or rm.get("usage", {}) or {}
    return (
        int(tu.get("prompt_tokens", 0) or 0),
        int(tu.get("completion_tokens", 0) or 0),
    )
