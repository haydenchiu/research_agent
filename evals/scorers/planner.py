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
    """LLM judge: are the sub-questions relevant and diverse? (binary rubric)"""

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
                        "You evaluate research sub-questions. For each criterion, "
                        "return 1 if the criterion is satisfied or 0 if not.\n"
                        "- relevance: do the questions address the research query?\n"
                        "- diversity: do the questions cover meaningfully different angles?\n"
                        "- specificity: are the questions specific enough to guide research?\n"
                        "Return JSON: {\"relevance\": int, \"diversity\": int, "
                        "\"specificity\": int, \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Research query: {research_query}\n\nSub-questions:\n{questions_text}",
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        relevance = int(bool(parsed.get("relevance", 0)))
        diversity = int(bool(parsed.get("diversity", 0)))
        specificity = int(bool(parsed.get("specificity", 0)))
        return {
            "relevance": relevance,
            "diversity": diversity,
            "specificity": specificity,
            "score": relevance + diversity + specificity,
            "explanation": parsed.get("explanation", ""),
        }


# ---------------------------------------------------------------------------
# LLM / With Ground Truth
# ---------------------------------------------------------------------------


class PlannerCoverageScorer(Scorer):
    """LLM judge: do the sub-questions cover the expected themes? (per-item hit count)"""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, target: dict, research_query: str) -> dict:
        questions = output.get("sub_questions", [])
        expected_themes = target.get("expected_themes", [])
        if not expected_themes:
            return {"theme_hits": 0, "theme_total": 0, "theme_coverage_llm": 1.0, "missing_themes": []}

        questions_text = "\n".join(f"- {q}" for q in questions)
        themes_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(expected_themes))

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate whether research sub-questions cover each expected theme.\n"
                        "For EACH theme, return 1 if the sub-questions address it or 0 if not.\n"
                        "Return JSON: {\"theme_results\": {\"<theme>\": 0 or 1, ...}, "
                        "\"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Sub-questions:\n{questions_text}\n\n"
                        f"Expected themes:\n{themes_list}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        theme_results = parsed.get("theme_results", {})
        hits = sum(int(bool(theme_results.get(t, 0))) for t in expected_themes)
        missing = [t for t in expected_themes if not theme_results.get(t, 0)]
        return {
            "theme_hits": hits,
            "theme_total": len(expected_themes),
            "theme_coverage_llm": hits / len(expected_themes),
            "missing_themes": missing,
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        planner_structure_check,
        planner_theme_overlap,
        PlannerRelevanceScorer(),
        PlannerCoverageScorer(),
    ]
