from __future__ import annotations

import weave
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import build_llm
from agents.utils import load_prompt, log_agent, parse_json_response
from graph.state import ResearchState


class FormatCheckerAgent(weave.Model):
    model_name: str = "gpt-4o-mini"
    provider: str = "openai"
    temperature: float = 0.1
    max_tokens: int = 4096

    @weave.op()
    def predict(self, draft_report: str) -> dict:
        llm = build_llm(self.provider, self.model_name, self.temperature, self.max_tokens)
        system_prompt = load_prompt("format_checker")

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Please review this research report:\n\n{draft_report}"),
            ]
        )

        parsed = parse_json_response(response.content)
        passed = parsed.get("passed", False)
        issues = parsed.get("issues", [])
        summary = parsed.get("summary", "")

        error_issues = [
            iss["description"]
            for iss in issues
            if isinstance(iss, dict) and iss.get("severity") == "error"
        ]

        checker_result = {
            "passed": passed,
            "issues": issues,
            "summary": summary,
        }

        if passed or not error_issues:
            return {"final_report": draft_report, "format_issues": [], "checker_result": checker_result}

        return {"final_report": "", "format_issues": error_issues, "checker_result": checker_result}

    def node(self, state: ResearchState) -> dict:
        draft = state.get("draft_report", "")
        format_revision_count = state.get("format_revision_count", 0)

        result = self.predict(draft_report=draft)

        if result.get("final_report"):
            result["format_revision_count"] = format_revision_count + 1
            result["messages"] = [log_agent("format_checker", "Report formatting approved")]
        else:
            result["format_revision_count"] = format_revision_count + 1
            result["messages"] = [
                log_agent(
                    "format_checker",
                    f"Found {len(result['format_issues'])} formatting error(s), sending back for revision",
                )
            ]
        return result
