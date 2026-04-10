from __future__ import annotations

import json
import re
from pathlib import Path

from config.settings import PROMPTS_DIR


def load_prompt(agent_name: str) -> str:
    """Load the system prompt for an agent from prompts/<agent_name>.md."""
    path = PROMPTS_DIR / f"{agent_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text()


def parse_json_response(text: str) -> dict:
    """Extract and parse JSON from an LLM response that may contain markdown fences."""
    # Try to find JSON in code fences first
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Fall back to finding the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from response:\n{text[:500]}")


def log_agent(agent_name: str, message: str) -> dict:
    """Create a message entry for the audit trail."""
    return {"agent": agent_name, "message": message}
