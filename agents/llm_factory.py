from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config.settings import AGENT_MODELS, ModelSpec, get_api_key


def get_llm(agent_name: str) -> BaseChatModel:
    """Return a LangChain chat model configured for the given agent."""
    spec = AGENT_MODELS.get(agent_name)
    if spec is None:
        raise ValueError(f"No model configured for agent: {agent_name}")
    return _build_llm(spec)


def _build_llm(spec: ModelSpec) -> BaseChatModel:
    api_key = get_api_key(spec.provider)
    if spec.provider == "openai":
        return ChatOpenAI(
            model=spec.model_name,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            api_key=api_key,
        )
    elif spec.provider == "anthropic":
        return ChatAnthropic(
            model=spec.model_name,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {spec.provider}")
