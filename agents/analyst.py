from __future__ import annotations

import weave
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import build_llm
from agents.utils import load_prompt, log_agent
from graph.state import ResearchState


class AnalystAgent(weave.Model):
    model_name: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    temperature: float = 0.3
    max_tokens: int = 4096

    @weave.op()
    def predict(self, research_query: str, search_results: list[dict]) -> dict:
        llm = build_llm(self.provider, self.model_name, self.temperature, self.max_tokens)
        system_prompt = load_prompt("analyst")

        results_text = _format_search_results(search_results)

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"Original research question: {research_query}\n\n"
                        f"Search findings:\n{results_text}"
                    )
                ),
            ]
        )

        return {"analysis": response.content}

    def node(self, state: ResearchState) -> dict:
        result = self.predict(
            research_query=state["research_query"],
            search_results=state.get("search_results", []),
        )
        result["messages"] = [log_agent("analyst", "Synthesized search results into analysis")]
        return result


def _format_search_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"### {r.get('title', 'Untitled')}")
        lines.append(f"Source: {r.get('url', 'N/A')}")
        lines.append(f"Query: {r.get('query', 'N/A')}")
        lines.append(f"Content: {r.get('content', '(no content)')}")
        lines.append("")
    return "\n".join(lines)
