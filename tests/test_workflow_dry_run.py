"""Dry-run test that exercises the full workflow with mocked LLM responses.

Validates:
- Graph topology (nodes, edges, conditional routing)
- State propagation between agents
- Output generation (Markdown + PDF)
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

from graph.state import ResearchState
from graph.workflow import build_workflow, compile_workflow


# ---------------------------------------------------------------------------
# Mock LLM responses for each agent
# ---------------------------------------------------------------------------

PLANNER_RESPONSE = json.dumps(
    {
        "sub_questions": [
            {"question": "What is the current state of AI adoption in workplaces?", "source_hint": "news"},
            {"question": "What jobs are most affected by AI automation?", "source_hint": "research"},
            {"question": "What are the counterarguments to AI job displacement?", "source_hint": "academic"},
        ]
    }
)

SEARCHER_RESPONSE = json.dumps(
    {
        "findings": [
            {
                "sub_question": "What is the current state of AI adoption?",
                "key_facts": ["AI adoption has grown 50% since 2020"],
                "sources": [{"url": "https://example.com/ai-report", "title": "AI Report 2025"}],
                "confidence": "high",
                "contradictions": [],
            }
        ]
    }
)

ANALYST_RESPONSE = textwrap.dedent(
    """\
    ## Analysis: Economic Impacts of AI on the Labor Market

    AI adoption has accelerated significantly since 2020, with roughly 50% growth
    in enterprise deployments [AI Report 2025](https://example.com/ai-report).

    ### Key Themes
    - Automation is concentrated in routine cognitive tasks.
    - New job categories are emerging alongside displacement.
    """
)

CRITIC_APPROVED_RESPONSE = json.dumps(
    {
        "approved": True,
        "overall_quality": "good",
        "gaps": [],
        "needs_data_analysis": False,
        "data_analysis_suggestion": "",
        "feedback": "Analysis is comprehensive and well-supported.",
        "additional_search_queries": [],
    }
)

WRITER_RESPONSE = textwrap.dedent(
    """\
    # Economic Impacts of AI on the Labor Market

    ## Executive Summary

    This report examines the economic impacts of artificial intelligence on employment.

    ## Introduction

    Artificial intelligence is transforming the global economy.

    ## Methodology

    Information was gathered through systematic web search and synthesis.

    ## Findings

    AI adoption has grown 50% since 2020, primarily in routine cognitive tasks.

    ## Discussion

    The evidence suggests a mixed picture with both displacement and creation effects.

    ## Conclusion

    AI's impact on labor markets is significant but nuanced.

    ## References

    1. [AI Report 2025](https://example.com/ai-report)
    """
)

FORMAT_CHECKER_PASS = json.dumps(
    {
        "passed": True,
        "issues": [],
        "summary": "Report meets all formatting requirements.",
    }
)


# ---------------------------------------------------------------------------
# Fake Tavily search
# ---------------------------------------------------------------------------

MOCK_SEARCH_RESULTS = [
    {
        "query": "AI adoption in workplaces",
        "url": "https://example.com/ai-report",
        "title": "AI Report 2025",
        "content": "AI adoption has grown 50% since 2020 across enterprises.",
        "relevance_score": 0.95,
    },
]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(response_text: str) -> MagicMock:
    """Create a mock LLM that returns a fixed response."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock.invoke.return_value = mock_response
    return mock


def _get_mock_llm_for_agent(agent_name: str) -> MagicMock:
    """Return the appropriate mock LLM for each agent."""
    response_map = {
        "planner": PLANNER_RESPONSE,
        "searcher": SEARCHER_RESPONSE,
        "analyst": ANALYST_RESPONSE,
        "critic": CRITIC_APPROVED_RESPONSE,
        "data_analyst": "{}",
        "chart_reviewer": '{"approved": true, "issues": [], "suggestions": []}',
        "writer": WRITER_RESPONSE,
        "format_checker": FORMAT_CHECKER_PASS,
    }
    return _make_mock_llm(response_map.get(agent_name, "{}"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_graph_topology():
    """Verify the graph has all expected nodes and compiles."""
    builder = build_workflow()
    graph = builder.compile()
    node_names = set(graph.nodes.keys())

    expected = {
        "__start__",
        "planner",
        "searcher",
        "analyst",
        "critic",
        "data_analyst",
        "chart_reviewer",
        "writer",
        "format_checker",
    }
    # __start__ is added by LangGraph
    assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"
    print("  [PASS] Graph topology has all 8 agent nodes")


def test_full_workflow_happy_path():
    """Run the full workflow with mocked LLMs (critic approves on first pass)."""
    output_dir = Path("output/test_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        patch("agents.planner.get_llm", lambda _: _get_mock_llm_for_agent("planner")),
        patch("agents.searcher.get_llm", lambda _: _get_mock_llm_for_agent("searcher")),
        patch("agents.analyst.get_llm", lambda _: _get_mock_llm_for_agent("analyst")),
        patch("agents.critic.get_llm", lambda _: _get_mock_llm_for_agent("critic")),
        patch("agents.writer.get_llm", lambda _: _get_mock_llm_for_agent("writer")),
        patch("agents.format_checker.get_llm", lambda _: _get_mock_llm_for_agent("format_checker")),
        patch("agents.searcher.tavily_search", return_value=MOCK_SEARCH_RESULTS),
    ):
        workflow = compile_workflow()
        initial_state = {
            "research_query": "What are the economic impacts of AI on the labor market?",
            "sub_questions": [],
            "search_results": [],
            "analysis": "",
            "critique": {},
            "data_analysis_needed": False,
            "data_analysis_result": "",
            "chart_paths": [],
            "chart_review": {},
            "chart_revision_count": 0,
            "draft_report": "",
            "format_issues": [],
            "final_report": "",
            "revision_count": 0,
            "max_revisions": 3,
            "format_revision_count": 0,
            "messages": [],
        }

        result = workflow.invoke(initial_state)

    # Verify final state
    assert result.get("final_report"), "No final report generated"
    assert "Economic Impacts" in result["final_report"]
    assert len(result.get("messages", [])) >= 5, "Expected messages from at least 5 agents"
    assert result.get("critique", {}).get("approved") is True

    # Verify agents that ran
    agents_seen = {m["agent"] for m in result.get("messages", [])}
    expected_agents = {"planner", "searcher", "analyst", "critic", "writer", "format_checker"}
    assert expected_agents.issubset(agents_seen), f"Missing agents: {expected_agents - agents_seen}"

    print(f"  [PASS] Full workflow completed, final report length: {len(result['final_report'])} chars")
    print(f"  [PASS] Agents executed: {sorted(agents_seen)}")

    # Test PDF export
    from tools.pdf_export import export_pdf

    pdf_path = export_pdf(result["final_report"], output_dir / "test_report.pdf")
    assert pdf_path.exists(), "PDF was not created"
    print(f"  [PASS] PDF exported: {pdf_path}")

    # Save markdown too
    md_path = output_dir / "test_report.md"
    md_path.write_text(result["final_report"])
    print(f"  [PASS] Markdown saved: {md_path}")

    # Cleanup
    shutil.rmtree(output_dir, ignore_errors=True)


def test_routing_logic():
    """Test the conditional routing functions directly."""
    from graph.workflow import (
        _route_after_chart_reviewer,
        _route_after_critic,
        _route_after_format_checker,
    )

    # Critic: approved -> writer
    state_approved = {"critique": {"approved": True}, "revision_count": 1, "max_revisions": 3}
    assert _route_after_critic(state_approved) == "writer"

    # Critic: not approved, has gaps -> searcher
    state_gaps = {
        "critique": {
            "approved": False,
            "gaps": ["Missing economic data"],
            "additional_search_queries": ["economic impact AI data"],
        },
        "revision_count": 1,
        "max_revisions": 3,
    }
    assert _route_after_critic(state_gaps) == "searcher"

    # Critic: needs data analysis -> data_analyst
    state_data = {
        "critique": {"approved": False, "needs_data_analysis": True},
        "revision_count": 1,
        "max_revisions": 3,
    }
    assert _route_after_critic(state_data) == "data_analyst"

    # Critic: max revisions exceeded -> writer
    state_max = {
        "critique": {"approved": False, "gaps": ["still missing"]},
        "revision_count": 3,
        "max_revisions": 3,
    }
    assert _route_after_critic(state_max) == "writer"

    # Chart reviewer: approved -> writer
    assert _route_after_chart_reviewer({"chart_review": {"approved": True}, "chart_revision_count": 0}) == "writer"

    # Chart reviewer: not approved, revisions left -> data_analyst
    assert (
        _route_after_chart_reviewer({"chart_review": {"approved": False}, "chart_revision_count": 0})
        == "data_analyst"
    )

    # Chart reviewer: not approved but max revisions -> writer
    assert _route_after_chart_reviewer({"chart_review": {"approved": False}, "chart_revision_count": 2}) == "writer"

    # Format checker: no issues -> end
    assert _route_after_format_checker({"format_issues": [], "format_revision_count": 1}) == "__end__"

    # Format checker: has issues -> writer
    assert _route_after_format_checker({"format_issues": ["bad heading"], "format_revision_count": 0}) == "writer"

    print("  [PASS] All routing logic tests passed")


if __name__ == "__main__":
    print("\nRunning dry-run tests...\n")
    test_graph_topology()
    test_routing_logic()
    test_full_workflow_happy_path()
    print("\nAll tests passed!")
