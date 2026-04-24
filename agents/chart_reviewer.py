from __future__ import annotations

import weave
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import build_llm
from agents.utils import load_prompt, log_agent, parse_json_response
from graph.state import ResearchState
from tools.chart_loader import build_image_message_content


class ChartReviewerAgent(weave.Model):
    model_name: str = "gpt-4o"
    provider: str = "openai"
    temperature: float = 0.2
    max_tokens: int = 4096

    @weave.op()
    def predict(
        self,
        chart_paths: list[str],
        data_analysis_result: str = "",
    ) -> dict:
        if not chart_paths:
            return {"chart_review": {"approved": True, "issues": [], "suggestions": []}}

        llm = build_llm(self.provider, self.model_name, self.temperature, self.max_tokens)
        system_prompt = load_prompt("chart_reviewer")

        image_content = build_image_message_content(chart_paths)
        text_content = {
            "type": "text",
            "text": (
                f"Please review the following {len(chart_paths)} chart(s) generated for a "
                f"research report. The data analysis context is:\n\n"
                f"{data_analysis_result[:1500]}"
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
        return {"chart_review": review}

    def node(self, state: ResearchState) -> dict:
        chart_paths = state.get("chart_paths", [])
        chart_revision_count = state.get("chart_revision_count", 0)

        result = self.predict(
            chart_paths=chart_paths,
            data_analysis_result=state.get("data_analysis_result", ""),
        )

        if not chart_paths:
            result["chart_revision_count"] = chart_revision_count
            result["messages"] = [log_agent("chart_reviewer", "No charts to review, auto-approved")]
            return result

        review = result["chart_review"]
        status = "approved" if review["approved"] else f"revision needed ({len(review['issues'])} issues)"

        result["chart_revision_count"] = chart_revision_count + 1
        result["messages"] = [log_agent("chart_reviewer", f"Chart review: {status}")]
        return result
