from __future__ import annotations

import base64
from pathlib import Path


def load_chart_as_base64(path: str) -> str:
    """Read a chart image file and return its base64-encoded content."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Chart image not found: {path}")
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def build_image_message_content(chart_paths: list[str]) -> list[dict]:
    """Build a list of LangChain-compatible image content blocks for vision models."""
    content: list[dict] = []
    for path in chart_paths:
        b64 = load_chart_as_base64(path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            }
        )
    return content
