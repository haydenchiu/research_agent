"""Analyst agent scorers."""

from __future__ import annotations

import re

import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

# ---------------------------------------------------------------------------
# Code / No Ground Truth
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+")
HEADER_PATTERN = re.compile(r"^#{1,6}\s+", re.MULTILINE)


@weave.op()
def analyst_structure_check(output: dict) -> dict:
    """Check minimum length, markdown headers, and citations."""
    analysis = output.get("analysis", "")
    has_min_length = len(analysis) >= 500
    has_headers = bool(HEADER_PATTERN.search(analysis))
    has_citations = bool(URL_PATTERN.search(analysis))
    return {
        "has_min_length": has_min_length,
        "has_headers": has_headers,
        "has_citations": has_citations,
    }


# ---------------------------------------------------------------------------
# Code / With Ground Truth
# ---------------------------------------------------------------------------


@weave.op()
def analyst_term_overlap(output: dict, target: dict) -> dict:
    """Check overlap of key terms with the expected analysis."""
    analysis = output.get("analysis", "").lower()
    expected_terms = target.get("expected_terms", [])
    if not expected_terms:
        return {"term_overlap": 1.0}

    hits = sum(1 for term in expected_terms if term.lower() in analysis)
    return {"term_overlap": hits / len(expected_terms)}


# ---------------------------------------------------------------------------
# LLM / No Ground Truth
# ---------------------------------------------------------------------------


class AnalystQualityScorer(Scorer):
    """LLM judge: coherence, depth, and balance of the analysis."""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, research_query: str) -> dict:
        analysis = output.get("analysis", "")

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate a research analysis on these criteria (1-5 each):\n"
                        "- coherence: logical flow and structure\n"
                        "- depth: thoroughness of coverage\n"
                        "- balance: fair representation of different perspectives\n"
                        "Return JSON: {\"coherence\": int, \"depth\": int, \"balance\": int, \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Research query: {research_query}\n\nAnalysis:\n{analysis[:3000]}",
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        return {
            "coherence": parsed.get("coherence", 3) / 5.0,
            "depth": parsed.get("depth", 3) / 5.0,
            "balance": parsed.get("balance", 3) / 5.0,
            "explanation": parsed.get("explanation", ""),
        }


# ---------------------------------------------------------------------------
# LLM / With Ground Truth
# ---------------------------------------------------------------------------


class AnalystFindingsCoverageScorer(Scorer):
    """LLM judge: does the analysis cover the gold-standard key findings?"""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, target: dict, research_query: str) -> dict:
        analysis = output.get("analysis", "")
        key_findings = target.get("key_findings", [])
        findings_text = "\n".join(f"- {f}" for f in key_findings)

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate whether the analysis covers all expected key findings.\n"
                        "Score 1-5. Return JSON: {\"coverage\": int, \"missing\": [str], \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Analysis:\n{analysis[:3000]}\n\n"
                        f"Expected key findings:\n{findings_text}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        return {
            "findings_coverage": parsed.get("coverage", 3) / 5.0,
            "missing_findings": parsed.get("missing", []),
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        analyst_structure_check,
        analyst_term_overlap,
        AnalystQualityScorer(),
        AnalystFindingsCoverageScorer(),
    ]
