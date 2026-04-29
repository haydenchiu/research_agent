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
    """LLM judge: quality of the data analysis summary (binary per criterion)."""

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
                        "Evaluate a data analysis summary. For each criterion, "
                        "return 1 if satisfied or 0 if not.\n"
                        "- clarity: the summary is easy to understand\n"
                        "- insight: the analysis provides meaningful, non-obvious findings\n"
                        "- data_grounding: claims are supported by specific data points\n"
                        "Return JSON: {\"clarity\": int, \"insight\": int, "
                        "\"data_grounding\": int, \"explanation\": str}"
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
        clarity = int(bool(parsed.get("clarity", 0)))
        insight = int(bool(parsed.get("insight", 0)))
        data_grounding = int(bool(parsed.get("data_grounding", 0)))
        return {
            "clarity": clarity,
            "insight": insight,
            "data_grounding": data_grounding,
            "score": clarity + insight + data_grounding,
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        data_analyst_structure_check,
        DataAnalystQualityScorer(),
    ]
