from __future__ import annotations

import weave
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import build_llm
from agents.utils import load_prompt, log_agent
from graph.state import ResearchState


class WriterAgent(weave.Model):
    model_name: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    temperature: float = 0.3
    max_tokens: int = 8192

    @weave.op()
    def predict(
        self,
        research_query: str,
        analysis: str,
        format_issues: list[str] | None = None,
    ) -> dict:
        llm = build_llm(self.provider, self.model_name, self.temperature, self.max_tokens)
        system_prompt = load_prompt("writer")
        format_issues = format_issues or []

        context_parts = [
            f"Original research question: {research_query}",
            f"\n## Analysis\n{analysis}",
        ]
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

        return {"draft_report": response.content}

    def node(self, state: ResearchState) -> dict:
        result = self.predict(
            research_query=state["research_query"],
            analysis=state.get("analysis", ""),
            format_issues=state.get("format_issues", []),
        )
        result["messages"] = [log_agent("writer", "Produced research report draft")]
        return result
