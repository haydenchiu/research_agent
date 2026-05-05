from __future__ import annotations

import weave
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import build_llm
from agents.utils import load_prompt, log_agent, parse_json_response
from graph.state import ResearchState


class CriticAgent(weave.Model):
    model_name: str = "gpt-4o"
    provider: str = "openai"
    temperature: float = 0.4
    max_tokens: int = 4096

    @weave.op()
    def predict(
        self,
        research_query: str,
        analysis: str,
        revision_count: int = 0,
        max_revisions: int = 3,
    ) -> dict:
        llm = build_llm(self.provider, self.model_name, self.temperature, self.max_tokens)
        system_prompt = load_prompt("critic")

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"Original research question: {research_query}\n\n"
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
            "feedback": parsed.get("feedback", ""),
            "additional_search_queries": parsed.get("additional_search_queries", []),
        }
        return {"critique": critique}

    def node(self, state: ResearchState) -> dict:
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 3)

        result = self.predict(
            research_query=state["research_query"],
            analysis=state.get("analysis", ""),
            revision_count=revision_count,
            max_revisions=max_revisions,
        )

        critique = result["critique"]

        new_sub_questions = state.get("sub_questions", [])
        if not critique["approved"] and revision_count < max_revisions:
            extra_queries = critique.get("additional_search_queries", [])
            if extra_queries:
                new_sub_questions = extra_queries

        status = "approved" if critique["approved"] else "revision requested"

        result["sub_questions"] = new_sub_questions
        result["revision_count"] = revision_count + 1
        result["messages"] = [log_agent("critic", f"Review: {status}")]
        return result
