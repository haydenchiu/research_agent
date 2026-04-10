from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"


@dataclass(frozen=True)
class ModelSpec:
    provider: str  # "openai" or "anthropic"
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 4096


AGENT_MODELS: dict[str, ModelSpec] = {
    "planner": ModelSpec(provider="anthropic", model_name="claude-sonnet-4-20250514"),
    "searcher": ModelSpec(provider="openai", model_name="gpt-4o-mini", temperature=0.1),
    "analyst": ModelSpec(provider="anthropic", model_name="claude-sonnet-4-20250514"),
    "critic": ModelSpec(provider="openai", model_name="gpt-4o", temperature=0.4),
    "data_analyst": ModelSpec(provider="openai", model_name="gpt-4o"),
    "chart_reviewer": ModelSpec(provider="openai", model_name="gpt-4o", temperature=0.2),
    "writer": ModelSpec(provider="anthropic", model_name="claude-sonnet-4-20250514", max_tokens=8192),
    "format_checker": ModelSpec(provider="openai", model_name="gpt-4o-mini", temperature=0.1),
}

DEFAULT_MAX_REVISIONS = 3
DEFAULT_MAX_CHART_REVISIONS = 2


def get_api_key(provider: str) -> str:
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_var = key_map.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    value = os.getenv(env_var)
    if not value:
        raise EnvironmentError(
            f"{env_var} not set. Copy .env.example to .env and fill in your keys."
        )
    return value


def get_tavily_api_key() -> str:
    value = os.getenv("TAVILY_API_KEY")
    if not value:
        raise EnvironmentError(
            "TAVILY_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return value
