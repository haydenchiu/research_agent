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
    """LLM judge: writing quality rubric (binary per criterion)."""

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
                        "Evaluate a research report. For each criterion, "
                        "return 1 if satisfied or 0 if not.\n"
                        "- coherence: sections flow logically from one to the next\n"
                        "- completeness: the topic is covered thoroughly without major gaps\n"
                        "- clarity: the writing is clear and understandable\n"
                        "- citations: references and sources are present and properly used\n"
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
        coherence = int(bool(parsed.get("coherence", 0)))
        completeness = int(bool(parsed.get("completeness", 0)))
        clarity = int(bool(parsed.get("clarity", 0)))
        citations = int(bool(parsed.get("citations", 0)))
        return {
            "coherence": coherence,
            "completeness": completeness,
            "clarity": clarity,
            "citations": citations,
            "score": coherence + completeness + clarity + citations,
            "explanation": parsed.get("explanation", ""),
        }


# ---------------------------------------------------------------------------
# LLM / With Ground Truth
# ---------------------------------------------------------------------------


class WriterTalkingPointsScorer(Scorer):
    """LLM judge: does the report cover the gold-standard talking points? (per-item hit count)"""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, target: dict, research_query: str) -> dict:
        report = output.get("draft_report", "")
        talking_points = target.get("talking_points", [])
        if not talking_points:
            return {"point_hits": 0, "point_total": 0, "talking_point_coverage": 1.0, "missing_points": []}

        points_list = "\n".join(f"{i+1}. {p}" for i, p in enumerate(talking_points))

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate whether a research report covers each expected talking point.\n"
                        "For EACH talking point, return 1 if the report addresses it or 0 if not.\n"
                        "Return JSON: {\"point_results\": {\"<talking point>\": 0 or 1, ...}, "
                        "\"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Report:\n{report[:4000]}\n\n"
                        f"Expected talking points:\n{points_list}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        point_results = parsed.get("point_results", {})
        hits = sum(int(bool(point_results.get(p, 0))) for p in talking_points)
        missing = [p for p in talking_points if not point_results.get(p, 0)]
        return {
            "point_hits": hits,
            "point_total": len(talking_points),
            "talking_point_coverage": hits / len(talking_points),
            "missing_points": missing,
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        writer_structure_check,
        writer_key_terms,
        WriterQualityScorer(),
        WriterTalkingPointsScorer(),
    ]
