"""
OpenAI Client Builder

Centralizes OpenAI client initialization logic to reduce code duplication and facilitate future replacement and injection (NodeAgent/strategy).
Behavior remains consistent with original: uses `llm.base_url` and `llm.api_key` from `config.config`.
"""

from typing import Any
from openai import OpenAI
from config.config import config


def build_openai_client() -> OpenAI:
    """Build and return an OpenAI client with parameters from global configuration."""
    return OpenAI(
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
    )