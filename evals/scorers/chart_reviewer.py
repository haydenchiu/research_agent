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
    """LLM judge: how thorough is the chart review? (binary per criterion)"""

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
                        "Evaluate a chart review for thoroughness. For each criterion, "
                        "return 1 if the review addressed it or 0 if not.\n"
                        "- axes_check: the review assessed axis labels and scales\n"
                        "- title_check: the review assessed chart titles\n"
                        "- readability: the review assessed visual clarity and readability\n"
                        "- data_accuracy: the review assessed whether the chart accurately represents the data\n"
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
        axes_check = int(bool(parsed.get("axes_check", 0)))
        title_check = int(bool(parsed.get("title_check", 0)))
        readability = int(bool(parsed.get("readability", 0)))
        data_accuracy = int(bool(parsed.get("data_accuracy", 0)))
        return {
            "axes_check": axes_check,
            "title_check": title_check,
            "readability": readability,
            "data_accuracy": data_accuracy,
            "score": axes_check + title_check + readability + data_accuracy,
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        chart_reviewer_structure_check,
        ChartReviewThoroughnessScorer(),
    ]
