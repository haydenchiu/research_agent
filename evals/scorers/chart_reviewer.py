"""Chart reviewer agent scorers."""

from __future__ import annotations

import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

# ---------------------------------------------------------------------------
# Code / No Ground Truth
# ---------------------------------------------------------------------------


@weave.op()
def chart_reviewer_structure_check(output: dict) -> dict:
    """Check that the chart review has valid structure."""
    review = output.get("chart_review", {})
    approved_is_bool = isinstance(review.get("approved"), bool)
    issues_is_list = isinstance(review.get("issues"), list)
    suggestions_is_list = isinstance(review.get("suggestions"), list)
    return {
        "approved_is_bool": approved_is_bool,
        "issues_is_list": issues_is_list,
        "suggestions_is_list": suggestions_is_list,
    }


# ---------------------------------------------------------------------------
# LLM / No Ground Truth
# ---------------------------------------------------------------------------


class ChartReviewThoroughnessScorer(Scorer):
    """LLM judge: how thorough is the chart review?"""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, data_analysis_result: str) -> dict:
        review = output.get("chart_review", {})
        issues = review.get("issues", [])
        suggestions = review.get("suggestions", [])

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate a chart review for thoroughness using this rubric (1-5 each):\n"
                        "- axes_check: did the review assess axis labels and scales?\n"
                        "- title_check: did it assess chart titles?\n"
                        "- readability: did it assess visual clarity and readability?\n"
                        "- data_accuracy: did it assess whether the chart accurately represents the data?\n"
                        "Return JSON: {\"axes_check\": int, \"title_check\": int, "
                        "\"readability\": int, \"data_accuracy\": int, \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Data analysis context:\n{data_analysis_result[:1500]}\n\n"
                        f"Review issues: {issues}\n"
                        f"Review suggestions: {suggestions}\n"
                        f"Approved: {review.get('approved')}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        return {
            "axes_check": parsed.get("axes_check", 3) / 5.0,
            "title_check": parsed.get("title_check", 3) / 5.0,
            "readability": parsed.get("readability", 3) / 5.0,
            "data_accuracy": parsed.get("data_accuracy", 3) / 5.0,
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        chart_reviewer_structure_check,
        ChartReviewThoroughnessScorer(),
    ]
