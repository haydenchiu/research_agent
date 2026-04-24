from __future__ import annotations

import weave
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import get_llm
from agents.utils import load_prompt, log_agent, parse_json_response
from graph.state import ResearchState


@weave.op
def planner_node(state: ResearchState) -> dict:
    """Decompose the research query into sub-questions."""
    llm = get_llm("planner")
    system_prompt = load_prompt("planner")
    query = state["research_query"]

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Research question: {query}"),
        ]
    )

    parsed = parse_json_response(response.content)
    sub_questions = [item["question"] for item in parsed.get("sub_questions", [])]

    return {
        "sub_questions": sub_questions,
        "messages": [
            log_agent("planner", f"Decomposed into {len(sub_questions)} sub-questions")
        ],
    }
