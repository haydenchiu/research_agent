from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.analyst import analyst_node
from agents.chart_reviewer import chart_reviewer_node
from agents.critic import critic_node
from agents.data_analyst import data_analyst_node
from agents.format_checker import format_checker_node
from agents.planner import planner_node
from agents.searcher import searcher_node
from agents.writer import writer_node
from config.settings import DEFAULT_MAX_CHART_REVISIONS, DEFAULT_MAX_REVISIONS
from graph.state import ResearchState


def _route_after_critic(
    state: ResearchState,
) -> Literal["searcher", "data_analyst", "writer"]:
    critique = state.get("critique", {})
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", DEFAULT_MAX_REVISIONS)

    if critique.get("needs_data_analysis"):
        return "data_analyst"

    if not critique.get("approved") and revision_count < max_revisions:
        gaps = critique.get("gaps", [])
        extra_queries = critique.get("additional_search_queries", [])
        if gaps or extra_queries:
            return "searcher"

    return "writer"


def _route_after_chart_reviewer(
    state: ResearchState,
) -> Literal["data_analyst", "writer"]:
    chart_review = state.get("chart_review", {})
    chart_revision_count = state.get("chart_revision_count", 0)

    if not chart_review.get("approved") and chart_revision_count < DEFAULT_MAX_CHART_REVISIONS:
        return "data_analyst"

    return "writer"


def _route_after_format_checker(
    state: ResearchState,
) -> Literal["writer", "__end__"]:
    format_issues = state.get("format_issues", [])
    format_revision_count = state.get("format_revision_count", 0)

    if format_issues and format_revision_count < 2:
        return "writer"

    return END


def build_workflow() -> StateGraph:
    """Construct the multi-agent research workflow graph."""
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_node)
    builder.add_node("searcher", searcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("critic", critic_node)
    builder.add_node("data_analyst", data_analyst_node)
    builder.add_node("chart_reviewer", chart_reviewer_node)
    builder.add_node("writer", writer_node)
    builder.add_node("format_checker", format_checker_node)

    # Linear edges
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "searcher")
    builder.add_edge("searcher", "analyst")
    builder.add_edge("analyst", "critic")

    # Conditional: Critic decides next step
    builder.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"searcher": "searcher", "data_analyst": "data_analyst", "writer": "writer"},
    )

    # Data Analyst always goes to Chart Reviewer
    builder.add_edge("data_analyst", "chart_reviewer")

    # Conditional: Chart Reviewer decides next step
    builder.add_conditional_edges(
        "chart_reviewer",
        _route_after_chart_reviewer,
        {"data_analyst": "data_analyst", "writer": "writer"},
    )

    # Writer always goes to Format Checker
    builder.add_edge("writer", "format_checker")

    # Conditional: Format Checker decides if done
    builder.add_conditional_edges(
        "format_checker",
        _route_after_format_checker,
        {"writer": "writer", END: END},
    )

    return builder


def compile_workflow():
    """Build and compile the research workflow graph."""
    builder = build_workflow()
    return builder.compile()
