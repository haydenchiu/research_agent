"""Writer agent scorers."""

from __future__ import annotations

import re

import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

# ---------------------------------------------------------------------------
# Code / No Ground Truth
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "Executive Summary",
    "Introduction",
    "Methodology",
    "Findings",
    "Discussion",
    "Conclusion",
    "References",
]

REFERENCE_PATTERN = re.compile(r"\[.*?\]\(https?://\S+\)")


@weave.op()
def writer_structure_check(output: dict) -> dict:
    """Check required sections, minimum length, and references."""
    report = output.get("draft_report", "")
    report_lower = report.lower()

    sections_present = sum(
        1 for section in REQUIRED_SECTIONS if section.lower() in report_lower
    )
    section_ratio = sections_present / len(REQUIRED_SECTIONS)

    has_min_length = len(report) >= 1000
    has_references = bool(REFERENCE_PATTERN.search(report))

    return {
        "section_coverage": section_ratio,
        "has_min_length": has_min_length,
        "has_references": has_references,
    }


# ---------------------------------------------------------------------------
# Code / With Ground Truth
# ---------------------------------------------------------------------------


@weave.op()
def writer_key_terms(output: dict, target: dict) -> dict:
    """Check that key terms from ground truth appear in the report."""
    report = output.get("draft_report", "").lower()
    key_terms = target.get("key_terms", [])
    if not key_terms:
        return {"key_term_coverage": 1.0}

    hits = sum(1 for term in key_terms if term.lower() in report)
    return {"key_term_coverage": hits / len(key_terms)}


# ---------------------------------------------------------------------------
# LLM / No Ground Truth
# ---------------------------------------------------------------------------


class WriterQualityScorer(Scorer):
    """LLM judge: writing quality rubric."""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, research_query: str) -> dict:
        report = output.get("draft_report", "")

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate a research report on these criteria (1-5 each):\n"
                        "- coherence: logical flow between sections\n"
                        "- completeness: thorough coverage of the topic\n"
                        "- clarity: clear and understandable writing\n"
                        "- citations: proper use of references and sources\n"
                        "Return JSON: {\"coherence\": int, \"completeness\": int, "
                        "\"clarity\": int, \"citations\": int, \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Research query: {research_query}\n\nReport:\n{report[:4000]}",
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        return {
            "coherence": parsed.get("coherence", 3) / 5.0,
            "completeness": parsed.get("completeness", 3) / 5.0,
            "clarity": parsed.get("clarity", 3) / 5.0,
            "citations": parsed.get("citations", 3) / 5.0,
            "explanation": parsed.get("explanation", ""),
        }


# ---------------------------------------------------------------------------
# LLM / With Ground Truth
# ---------------------------------------------------------------------------


class WriterTalkingPointsScorer(Scorer):
    """LLM judge: does the report cover the gold-standard talking points?"""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, target: dict, research_query: str) -> dict:
        report = output.get("draft_report", "")
        talking_points = target.get("talking_points", [])
        points_text = "\n".join(f"- {p}" for p in talking_points)

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate whether a research report covers the expected talking points.\n"
                        "Score 1-5. Return JSON: {\"coverage\": int, \"missing_points\": [str], \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Report:\n{report[:4000]}\n\n"
                        f"Expected talking points:\n{points_text}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        return {
            "talking_point_coverage": parsed.get("coverage", 3) / 5.0,
            "missing_points": parsed.get("missing_points", []),
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        writer_structure_check,
        writer_key_terms,
        WriterQualityScorer(),
        WriterTalkingPointsScorer(),
    ]
