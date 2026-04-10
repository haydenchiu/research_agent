from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import get_llm
from agents.utils import load_prompt, log_agent
from graph.state import ResearchState


def writer_node(state: ResearchState) -> dict:
    """Produce the final structured research report."""
    llm = get_llm("writer")
    system_prompt = load_prompt("writer")

    query = state["research_query"]
    analysis = state.get("analysis", "")
    data_result = state.get("data_analysis_result", "")
    chart_paths = state.get("chart_paths", [])
    format_issues = state.get("format_issues", [])

    context_parts = [
        f"Original research question: {query}",
        f"\n## Analysis\n{analysis}",
    ]
    if data_result:
        context_parts.append(f"\n## Data Analysis Results\n{data_result}")
    if chart_paths:
        chart_refs = "\n".join(f"- ![Chart {i+1}]({p})" for i, p in enumerate(chart_paths))
        context_parts.append(f"\n## Available Charts\n{chart_refs}")
    if format_issues:
        issues_text = "\n".join(f"- {iss}" for iss in format_issues)
        context_parts.append(
            f"\n## Format Issues to Address\nPlease fix these issues from the previous draft:\n{issues_text}"
        )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(context_parts)),
        ]
    )

    return {
        "draft_report": response.content,
        "messages": [log_agent("writer", "Produced research report draft")],
    }
