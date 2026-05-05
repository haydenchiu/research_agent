from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict


@dataclass
class SearchResult:
    query: str
    url: str
    title: str
    content: str
    relevance_score: float = 0.0


@dataclass
class CritiqueResult:
    approved: bool
    gaps: list[str] = field(default_factory=list)
    feedback: str = ""



def _replace(existing: Any, new: Any) -> Any:
    """Reducer that always takes the newer value."""
    return new


def _append_lists(existing: list, new: list) -> list:
    """Reducer that concatenates lists."""
    return (existing or []) + (new or [])


class ResearchState(TypedDict, total=False):
    research_query: Annotated[str, _replace]
    sub_questions: Annotated[list[str], _replace]
    search_results: Annotated[list[dict], _append_lists]
    analysis: Annotated[str, _replace]
    critique: Annotated[dict, _replace]
    draft_report: Annotated[str, _replace]
    format_issues: Annotated[list[str], _replace]
    final_report: Annotated[str, _replace]
    revision_count: Annotated[int, _replace]
    max_revisions: Annotated[int, _replace]
    format_revision_count: Annotated[int, _replace]
    messages: Annotated[list[dict], _append_lists]
