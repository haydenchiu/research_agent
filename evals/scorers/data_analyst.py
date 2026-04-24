"""Data analyst agent scorers."""

from __future__ import annotations

import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

# ---------------------------------------------------------------------------
# Code / No Ground Truth
# ---------------------------------------------------------------------------


@weave.op()
def data_analyst_structure_check(output: dict) -> dict:
    """Check that the data analysis executed without errors and produced output."""
    result_text = output.get("data_analysis_result", "")
    chart_paths = output.get("chart_paths", [])
    has_summary = bool(result_text.strip())
    has_no_error = "**Error**" not in result_text
    has_charts = len(chart_paths) > 0
    return {
        "has_summary": has_summary,
        "execution_succeeded": has_no_error,
        "produced_charts": has_charts,
    }


# ---------------------------------------------------------------------------
# LLM / No Ground Truth
# ---------------------------------------------------------------------------


class DataAnalystQualityScorer(Scorer):
    """LLM judge: quality of the data analysis summary."""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, research_query: str) -> dict:
        result_text = output.get("data_analysis_result", "")

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate a data analysis summary on these criteria (1-5 each):\n"
                        "- clarity: is the summary easy to understand?\n"
                        "- insight: does it provide meaningful findings?\n"
                        "- data_grounding: are claims supported by data?\n"
                        "Return JSON: {\"clarity\": int, \"insight\": int, \"data_grounding\": int, \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Data analysis result:\n{result_text[:3000]}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        return {
            "clarity": parsed.get("clarity", 3) / 5.0,
            "insight": parsed.get("insight", 3) / 5.0,
            "data_grounding": parsed.get("data_grounding", 3) / 5.0,
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        data_analyst_structure_check,
        DataAnalystQualityScorer(),
    ]
