from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.settings import get_api_key


def build_llm(
    provider: str,
    model_name: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Build a LangChain chat model from explicit parameters."""
    api_key = get_api_key(provider)
    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
