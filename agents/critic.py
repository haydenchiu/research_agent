from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import get_llm
from agents.utils import load_prompt, log_agent, parse_json_response
from graph.state import ResearchState


def critic_node(state: ResearchState) -> dict:
    """Review the analysis for quality, completeness, and accuracy."""
    llm = get_llm("critic")
    system_prompt = load_prompt("critic")

    query = state["research_query"]
    analysis = state.get("analysis", "")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Original research question: {query}\n\n"
                    f"Analysis to review (revision {revision_count}/{max_revisions}):\n\n"
                    f"{analysis}"
                )
            ),
        ]
    )

    parsed = parse_json_response(response.content)

    critique = {
        "approved": parsed.get("approved", False),
        "gaps": parsed.get("gaps", []),
        "needs_data_analysis": parsed.get("needs_data_analysis", False),
        "feedback": parsed.get("feedback", ""),
        "additional_search_queries": parsed.get("additional_search_queries", []),
    }

    # If not approved and we still have revisions, add the new queries as sub-questions
    new_sub_questions = state.get("sub_questions", [])
    if not critique["approved"] and revision_count < max_revisions:
        extra_queries = critique.get("additional_search_queries", [])
        if extra_queries:
            new_sub_questions = extra_queries

    status = "approved" if critique["approved"] else "revision requested"
    data_flag = " (data analysis needed)" if critique["needs_data_analysis"] else ""

    return {
        "critique": critique,
        "data_analysis_needed": critique["needs_data_analysis"],
        "sub_questions": new_sub_questions,
        "revision_count": revision_count + 1,
        "messages": [log_agent("critic", f"Review: {status}{data_flag}")],
    }
