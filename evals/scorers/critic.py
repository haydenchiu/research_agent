"""Critic agent scorers."""

from __future__ import annotations

import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

# ---------------------------------------------------------------------------
# Code / No Ground Truth
# ---------------------------------------------------------------------------


@weave.op()
def critic_structure_check(output: dict) -> dict:
    """Check that the critique has valid structure."""
    critique = output.get("critique", {})
    approved_is_bool = isinstance(critique.get("approved"), bool)
    gaps_is_list = isinstance(critique.get("gaps"), list)
    has_feedback = bool(critique.get("feedback", "").strip())
    return {
        "approved_is_bool": approved_is_bool,
        "gaps_is_list": gaps_is_list,
        "has_feedback": has_feedback,
    }


# ---------------------------------------------------------------------------
# Code / With Ground Truth
# ---------------------------------------------------------------------------


@weave.op()
def critic_verdict_accuracy(output: dict, target: dict) -> dict:
    """Check whether the critic's approval matches the expected verdict."""
    critique = output.get("critique", {})
    expected_approved = target.get("expected_approved")
    if expected_approved is None:
        return {"verdict_correct": True}
    return {"verdict_correct": critique.get("approved") == expected_approved}


# ---------------------------------------------------------------------------
# LLM / No Ground Truth
# ---------------------------------------------------------------------------


class CriticFeedbackQualityScorer(Scorer):
    """LLM judge: is the feedback specific and actionable?"""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, research_query: str, analysis: str) -> dict:
        critique = output.get("critique", {})
        feedback = critique.get("feedback", "")
        gaps = critique.get("gaps", [])

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate a research critique on these criteria (1-5 each):\n"
                        "- specificity: does it identify concrete issues?\n"
                        "- actionability: does it suggest clear next steps?\n"
                        "Return JSON: {\"specificity\": int, \"actionability\": int, \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n"
                        f"Analysis (excerpt): {analysis[:1000]}\n\n"
                        f"Critique feedback: {feedback}\n"
                        f"Gaps identified: {gaps}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        return {
            "specificity": parsed.get("specificity", 3) / 5.0,
            "actionability": parsed.get("actionability", 3) / 5.0,
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        critic_structure_check,
        critic_verdict_accuracy,
        CriticFeedbackQualityScorer(),
    ]
