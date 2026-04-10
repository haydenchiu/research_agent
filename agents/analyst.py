from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import get_llm
from agents.utils import load_prompt, log_agent
from graph.state import ResearchState


def analyst_node(state: ResearchState) -> dict:
    """Synthesize search results into a coherent analysis."""
    llm = get_llm("analyst")
    system_prompt = load_prompt("analyst")

    query = state["research_query"]
    search_results = state.get("search_results", [])

    results_text = _format_search_results(search_results)

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Original research question: {query}\n\n"
                    f"Search findings:\n{results_text}"
                )
            ),
        ]
    )

    return {
        "analysis": response.content,
        "messages": [log_agent("analyst", "Synthesized search results into analysis")],
    }


def _format_search_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"### {r.get('title', 'Untitled')}")
        lines.append(f"Source: {r.get('url', 'N/A')}")
        lines.append(f"Query: {r.get('query', 'N/A')}")
        lines.append(f"Content: {r.get('content', '(no content)')}")
        lines.append("")
    return "\n".join(lines)
