from __future__ import annotations

import weave
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import build_llm
from agents.utils import load_prompt, log_agent, parse_json_response
from graph.state import ResearchState
from tools.python_exec import execute_python


class DataAnalystAgent(weave.Model):
    model_name: str = "gpt-4o"
    provider: str = "openai"
    temperature: float = 0.3
    max_tokens: int = 4096

    @weave.op()
    def predict(
        self,
        research_query: str,
        analysis: str,
        critique: dict | None = None,
        chart_review: dict | None = None,
    ) -> dict:
        llm = build_llm(self.provider, self.model_name, self.temperature, self.max_tokens)
        system_prompt = load_prompt("data_analyst")
        critique = critique or {}
        chart_review = chart_review or {}

        data_suggestion = critique.get("data_analysis_suggestion", "")

        context_parts = [
            f"Original research question: {research_query}",
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
        }

    def node(self, state: ResearchState) -> dict:
        result = self.predict(
            research_query=state["research_query"],
            analysis=state.get("analysis", ""),
            critique=state.get("critique", {}),
            chart_review=state.get("chart_review", {}),
        )
        result["messages"] = [
            log_agent(
                "data_analyst",
                f"Executed analysis code, produced {len(result['chart_paths'])} chart(s)",
            )
        ]
        return result
