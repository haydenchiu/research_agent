"""Planner agent scorers (all 4 quadrants of the 2x2 matrix)."""

from __future__ import annotations

import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

# ---------------------------------------------------------------------------
# Code / No Ground Truth
# ---------------------------------------------------------------------------


@weave.op()
def planner_structure_check(output: dict) -> dict:
    """Check that the planner produced well-formed sub-questions."""
    questions = output.get("sub_questions", [])
    has_enough = len(questions) >= 3
    all_non_empty = all(isinstance(q, str) and len(q.strip()) > 0 for q in questions)
    no_duplicates = len(questions) == len(set(questions))
    return {
        "has_enough_questions": has_enough,
        "all_non_empty": all_non_empty,
        "no_duplicates": no_duplicates,
    }


# ---------------------------------------------------------------------------
# Code / With Ground Truth
# ---------------------------------------------------------------------------


@weave.op()
def planner_theme_overlap(output: dict, target: dict) -> dict:
    """Check keyword overlap between generated questions and expected themes."""
    questions = output.get("sub_questions", [])
    expected_themes = target.get("expected_themes", [])
    if not expected_themes:
        return {"theme_coverage": 1.0}

    questions_lower = " ".join(questions).lower()
    hits = sum(1 for theme in expected_themes if theme.lower() in questions_lower)
    coverage = hits / len(expected_themes)
    return {"theme_coverage": coverage}


# ---------------------------------------------------------------------------
# LLM / No Ground Truth
# ---------------------------------------------------------------------------


class PlannerRelevanceScorer(Scorer):
    """LLM judge: are the sub-questions relevant and diverse?"""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, research_query: str) -> dict:
        questions = output.get("sub_questions", [])
        questions_text = "\n".join(f"- {q}" for q in questions)

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate research sub-questions. Score from 1-5 on:\n"
                        "- relevance: do questions address the research query?\n"
                        "- diversity: do questions cover different angles?\n"
                        "Return JSON: {\"relevance\": int, \"diversity\": int, \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Research query: {research_query}\n\nSub-questions:\n{questions_text}",
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        relevance = parsed.get("relevance", 3) / 5.0
        diversity = parsed.get("diversity", 3) / 5.0
        return {
            "relevance": relevance,
            "diversity": diversity,
            "explanation": parsed.get("explanation", ""),
        }


# ---------------------------------------------------------------------------
# LLM / With Ground Truth
# ---------------------------------------------------------------------------


class PlannerCoverageScorer(Scorer):
    """LLM judge: do the sub-questions cover the expected themes?"""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, target: dict, research_query: str) -> dict:
        questions = output.get("sub_questions", [])
        expected_themes = target.get("expected_themes", [])
        questions_text = "\n".join(f"- {q}" for q in questions)
        themes_text = "\n".join(f"- {t}" for t in expected_themes)

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate whether research sub-questions cover expected themes.\n"
                        "Score from 1-5 how well the questions address all expected themes.\n"
                        "Return JSON: {\"coverage_score\": int, \"missing_themes\": [str], \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Sub-questions:\n{questions_text}\n\n"
                        f"Expected themes:\n{themes_text}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        return {
            "theme_coverage_llm": parsed.get("coverage_score", 3) / 5.0,
            "missing_themes": parsed.get("missing_themes", []),
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        planner_structure_check,
        planner_theme_overlap,
        PlannerRelevanceScorer(),
        PlannerCoverageScorer(),
    ]
