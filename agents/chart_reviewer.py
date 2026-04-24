from __future__ import annotations

import weave
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import get_llm
from agents.utils import load_prompt, log_agent, parse_json_response
from graph.state import ResearchState
from tools.chart_loader import build_image_message_content


@weave.op
def chart_reviewer_node(state: ResearchState) -> dict:
    """Visually inspect generated charts using a vision model."""
    llm = get_llm("chart_reviewer")
    system_prompt = load_prompt("chart_reviewer")
    chart_paths = state.get("chart_paths", [])
    chart_revision_count = state.get("chart_revision_count", 0)

    # If no charts were generated, auto-approve
    if not chart_paths:
        return {
            "chart_review": {"approved": True, "issues": [], "suggestions": []},
            "chart_revision_count": chart_revision_count,
            "messages": [log_agent("chart_reviewer", "No charts to review, auto-approved")],
        }

    # Build multimodal message with chart images
    image_content = build_image_message_content(chart_paths)
    text_content = {
        "type": "text",
        "text": (
            f"Please review the following {len(chart_paths)} chart(s) generated for a "
            f"research report. The data analysis context is:\n\n"
            f"{state.get('data_analysis_result', '(not available)')[:1500]}"
        ),
    }

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[text_content] + image_content),
        ]
    )

    parsed = parse_json_response(response.content)
    review = {
        "approved": parsed.get("approved", False),
        "issues": parsed.get("issues", []),
        "suggestions": parsed.get("suggestions", []),
        "feedback_for_regeneration": parsed.get("feedback_for_regeneration", ""),
    }

    status = "approved" if review["approved"] else f"revision needed ({len(review['issues'])} issues)"

    return {
        "chart_review": review,
        "chart_revision_count": chart_revision_count + 1,
        "messages": [log_agent("chart_reviewer", f"Chart review: {status}")],
    }
