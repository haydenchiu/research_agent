from __future__ import annotations

import operator
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
    needs_data_analysis: bool = False
    feedback: str = ""


@dataclass
class ChartReviewResult:
    approved: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


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
    data_analysis_needed: Annotated[bool, _replace]
    data_analysis_result: Annotated[str, _replace]
    chart_paths: Annotated[list[str], _replace]
    chart_review: Annotated[dict, _replace]
    chart_revision_count: Annotated[int, _replace]
    draft_report: Annotated[str, _replace]
    format_issues: Annotated[list[str], _replace]
    final_report: Annotated[str, _replace]
    revision_count: Annotated[int, _replace]
    max_revisions: Annotated[int, _replace]
    format_revision_count: Annotated[int, _replace]
    messages: Annotated[list[dict], _append_lists]
