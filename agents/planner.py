from __future__ import annotations

import weave
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import build_llm
from agents.utils import load_prompt, log_agent, parse_json_response
from graph.state import ResearchState


class PlannerAgent(weave.Model):
    model_name: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"
    temperature: float = 0.3
    max_tokens: int = 4096

    @weave.op()
    def predict(self, research_query: str) -> dict:
        """Return structured sub-questions with question and source_hint fields."""
        llm = build_llm(self.provider, self.model_name, self.temperature, self.max_tokens)
        system_prompt = load_prompt("planner")

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Research question: {research_query}"),
            ]
        )

        parsed = parse_json_response(response.content)
        sub_questions = parsed.get("sub_questions", [])
        return {"sub_questions": sub_questions}

    def node(self, state: ResearchState) -> dict:
        result = self.predict(research_query=state["research_query"])
        questions_flat = [
            item["question"] for item in result["sub_questions"]
        ]
        return {
            "sub_questions": questions_flat,
            "messages": [
                log_agent("planner", f"Decomposed into {len(questions_flat)} sub-questions")
            ],
        }
