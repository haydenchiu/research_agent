from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import get_llm
from agents.utils import load_prompt, log_agent, parse_json_response
from graph.state import ResearchState
from tools.python_exec import execute_python


def data_analyst_node(state: ResearchState) -> dict:
    """Generate and execute data analysis code when quantitative work is needed."""
    llm = get_llm("data_analyst")
    system_prompt = load_prompt("data_analyst")

    query = state["research_query"]
    analysis = state.get("analysis", "")
    critique = state.get("critique", {})
    data_suggestion = critique.get("data_analysis_suggestion", "")
    chart_review = state.get("chart_review", {})

    # Build context for the LLM
    context_parts = [
        f"Original research question: {query}",
        f"\nAnalysis summary:\n{analysis[:2000]}",
    ]
    if data_suggestion:
        context_parts.append(f"\nRequested data analysis: {data_suggestion}")
    if chart_review and not chart_review.get("approved", True):
        context_parts.append(
            f"\nChart revision feedback: {chart_review.get('feedback_for_regeneration', '')}"
        )
        context_parts.append(f"Issues: {chart_review.get('issues', [])}")
        context_parts.append(f"Suggestions: {chart_review.get('suggestions', [])}")

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(context_parts)),
        ]
    )

    parsed = parse_json_response(response.content)
    code = parsed.get("code", "")
    summary = parsed.get("summary", "")

    exec_result = execute_python(code)

    result_text_parts = [f"**Analysis Summary**: {summary}"]
    if exec_result["stdout"]:
        result_text_parts.append(f"\n**Output**:\n```\n{exec_result['stdout']}\n```")
    if exec_result["error"]:
        result_text_parts.append(f"\n**Error**:\n```\n{exec_result['error']}\n```")

    chart_paths = exec_result.get("charts", [])

    return {
        "data_analysis_result": "\n".join(result_text_parts),
        "chart_paths": chart_paths,
        "messages": [
            log_agent(
                "data_analyst",
                f"Executed analysis code, produced {len(chart_paths)} chart(s)",
            )
        ],
    }
