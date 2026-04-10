from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import get_llm
from agents.utils import load_prompt, log_agent, parse_json_response
from graph.state import ResearchState


def format_checker_node(state: ResearchState) -> dict:
    """Validate the Markdown structure and formatting of the report."""
    llm = get_llm("format_checker")
    system_prompt = load_prompt("format_checker")
    draft = state.get("draft_report", "")
    format_revision_count = state.get("format_revision_count", 0)

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Please review this research report:\n\n{draft}"),
        ]
    )

    parsed = parse_json_response(response.content)
    passed = parsed.get("passed", False)
    issues = parsed.get("issues", [])

    error_issues = [
        iss["description"]
        for iss in issues
        if isinstance(iss, dict) and iss.get("severity") == "error"
    ]

    if passed or not error_issues:
        return {
            "final_report": draft,
            "format_issues": [],
            "format_revision_count": format_revision_count + 1,
            "messages": [log_agent("format_checker", "Report formatting approved")],
        }

    return {
        "format_issues": error_issues,
        "format_revision_count": format_revision_count + 1,
        "messages": [
            log_agent(
                "format_checker",
                f"Found {len(error_issues)} formatting error(s), sending back for revision",
            )
        ],
    }
