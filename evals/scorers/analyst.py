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
    """LLM judge: coherence, depth, and balance of the analysis (binary per criterion)."""

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
                        "Evaluate a research analysis. For each criterion, "
                        "return 1 if satisfied or 0 if not.\n"
                        "- coherence: the analysis has logical flow and clear structure\n"
                        "- depth: the topic is covered thoroughly, not superficially\n"
                        "- balance: different perspectives are fairly represented\n"
                        "Return JSON: {\"coherence\": int, \"depth\": int, "
                        "\"balance\": int, \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Research query: {research_query}\n\nAnalysis:\n{analysis[:3000]}",
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        coherence = int(bool(parsed.get("coherence", 0)))
        depth = int(bool(parsed.get("depth", 0)))
        balance = int(bool(parsed.get("balance", 0)))
        return {
            "coherence": coherence,
            "depth": depth,
            "balance": balance,
            "score": coherence + depth + balance,
            "explanation": parsed.get("explanation", ""),
        }


# ---------------------------------------------------------------------------
# LLM / With Ground Truth
# ---------------------------------------------------------------------------


class AnalystFindingsCoverageScorer(Scorer):
    """LLM judge: does the analysis cover the gold-standard key findings? (per-item hit count)"""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, target: dict, research_query: str) -> dict:
        analysis = output.get("analysis", "")
        key_findings = target.get("key_findings", [])
        if not key_findings:
            return {"finding_hits": 0, "finding_total": 0, "findings_coverage": 1.0, "missing_findings": []}

        findings_list = "\n".join(f"{i+1}. {f}" for i, f in enumerate(key_findings))

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate whether the analysis covers each expected key finding.\n"
                        "For EACH finding, return 1 if the analysis addresses it or 0 if not.\n"
                        "Return JSON: {\"finding_results\": {\"<finding>\": 0 or 1, ...}, "
                        "\"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Analysis:\n{analysis[:3000]}\n\n"
                        f"Expected key findings:\n{findings_list}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        finding_results = parsed.get("finding_results", {})
        hits = sum(int(bool(finding_results.get(f, 0))) for f in key_findings)
        missing = [f for f in key_findings if not finding_results.get(f, 0)]
        return {
            "finding_hits": hits,
            "finding_total": len(key_findings),
            "findings_coverage": hits / len(key_findings),
            "missing_findings": missing,
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        analyst_structure_check,
        analyst_term_overlap,
        AnalystQualityScorer(),
        AnalystFindingsCoverageScorer(),
    ]
