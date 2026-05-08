"""Analyst agent scorers (all 4 quadrants of the 2x2 eval matrix).

1. W/ GT  & Code  – Key-term overlap
2. W/o GT & Code  – Inline citation density + structural integrity
3. W/ GT  & LLM   – Key-findings coverage
4. W/o GT & LLM   – 4-point binary rubric (thematic synthesis, nuance, tone, gaps)
"""

from __future__ import annotations

import re

import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
CITATION_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
HEADER_PATTERN = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)


def _extract_headers(text: str) -> list[str]:
    """Return lowercased header text for every markdown heading in *text*."""
    return [m.group(1).strip().lower() for m in HEADER_PATTERN.finditer(text)]


# ---------------------------------------------------------------------------
# 2. W/o GT & Code – Inline Citation Density + Structural Integrity
# ---------------------------------------------------------------------------

_CITATION_WINDOW = 3  # at least 1 citation per N sentences


@weave.op()
def analyst_citation_density(output: dict) -> dict:
    """Regex check: for every N sentences there is at least one [Source Title](URL) citation."""
    analysis = output.get("analysis", "")
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(analysis) if s.strip()]
    if not sentences:
        return {"citation_density_pass": False, "windows_checked": 0, "windows_passed": 0}

    windows_checked = 0
    windows_passed = 0
    for start in range(0, len(sentences), _CITATION_WINDOW):
        window = " ".join(sentences[start : start + _CITATION_WINDOW])
        windows_checked += 1
        if CITATION_PATTERN.search(window):
            windows_passed += 1

    all_pass = windows_passed == windows_checked
    return {
        "citation_density_pass": all_pass,
        "windows_checked": windows_checked,
        "windows_passed": windows_passed,
        "citation_density_ratio": windows_passed / windows_checked if windows_checked else 0.0,
    }


@weave.op()
def analyst_structural_integrity(output: dict) -> dict:
    """Verify mandatory sections: one about gaps in evidence and one about consensus/disagreement."""
    analysis = output.get("analysis", "")
    headers = _extract_headers(analysis)
    full_text_lower = analysis.lower()

    has_gaps_section = any("gap" in h for h in headers) or "gaps in evidence" in full_text_lower
    has_consensus_section = (
        any("consensus" in h or "disagreement" in h for h in headers)
        or "consensus and disagreement" in full_text_lower
    )

    return {
        "has_gaps_section": has_gaps_section,
        "has_consensus_section": has_consensus_section,
        "structural_integrity_pass": has_gaps_section and has_consensus_section,
    }


# ---------------------------------------------------------------------------
# 1. W/ GT & Code – Key-term overlap
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
# 4. W/o GT & LLM – 4-point binary rubric
# ---------------------------------------------------------------------------

_RUBRIC_CRITERIA = [
    "thematic_synthesis",
    "nuance_and_conflict",
    "objective_tone",
    "gap_identification",
]


class AnalystRubricScorer(Scorer):
    """LLM judge: 4-point binary rubric (no ground truth needed).

    1. Thematic Synthesis  – organised by themes, not by sub-question
    2. Nuance & Conflict   – explicitly contrasts at least two sources
    3. Objective Tone      – neutral, evidence-based, no fluff or opinion
    4. Gap Identification  – specific recommendation for missing information
    """

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, research_query: str) -> dict:
        analysis = output.get("analysis", "")

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate a research analysis against a 4-criterion "
                        "binary rubric. For each criterion return exactly 1 (pass) "
                        "or 0 (fail).\n\n"
                        "Criteria:\n"
                        "1. thematic_synthesis – The analysis is organized by broad "
                        "themes or topics, NOT structured around individual "
                        "sub-questions. The response should read as a unified "
                        "narrative rather than a list of question-by-question "
                        "answers.\n"
                        "2. nuance_and_conflict – The analysis explicitly contrasts "
                        "at least two different sources or viewpoints (e.g. "
                        "'While Source A claims X, Source B suggests Y').\n"
                        "3. objective_tone – The language is neutral and "
                        "evidence-based throughout. There is no promotional "
                        "language, personal opinion, or unsupported 'fluff'.\n"
                        "4. gap_identification – The analysis provides at least one "
                        "specific, logical recommendation for what information is "
                        "missing based on the findings presented.\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "thematic_synthesis": 0 or 1,\n'
                        '  "nuance_and_conflict": 0 or 1,\n'
                        '  "objective_tone": 0 or 1,\n'
                        '  "gap_identification": 0 or 1,\n'
                        '  "explanation": "<brief reasoning for each criterion>"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Analysis:\n{analysis[:3000]}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        scores = {c: int(bool(parsed.get(c, 0))) for c in _RUBRIC_CRITERIA}
        total = sum(scores.values())

        return {
            **scores,
            "rubric_total": total,
            "rubric_score": total / len(_RUBRIC_CRITERIA),
            "explanation": parsed.get("explanation", ""),
        }


# ---------------------------------------------------------------------------
# 3. W/ GT & LLM – Key-findings coverage
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
        analyst_term_overlap,            # 1. W/ GT  & Code
        analyst_citation_density,        # 2. W/o GT & Code
        analyst_structural_integrity,    # 2. W/o GT & Code
        AnalystFindingsCoverageScorer(), # 3. W/ GT  & LLM
        AnalystRubricScorer(),           # 4. W/o GT & LLM
    ]
