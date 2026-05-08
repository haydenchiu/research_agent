"""Writer agent scorers (all 4 quadrants of the 2x2 eval matrix).

1. W/ GT  & Code  -- Citation mapping accuracy, key metric retention, reference link integrity
2. W/o GT & Code  -- Section completeness, word count, markdown audit, transition density
3. W/ GT  & LLM   -- Tone alignment, synthesis fidelity
4. W/o GT & LLM   -- 4-point binary rubric (exec summary, thematic cohesion, citation
                      consistency, implications depth)
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
_REFERENCE_LINK_RE = re.compile(
    r"^\s*(\d+)\.\s*\[([^\]]*)\]\((https?://[^\s)]+)\)",
    re.MULTILINE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")

REQUIRED_SECTIONS = [
    "Title",
    "Executive Summary",
    "Introduction",
    "Methodology",
    "Findings",
    "Discussion",
    "Conclusion",
    "References",
]

TRANSITION_WORDS = [
    "furthermore", "moreover", "additionally", "in addition",
    "however", "nevertheless", "nonetheless", "on the other hand",
    "in contrast", "conversely", "whereas",
    "consequently", "therefore", "thus", "as a result", "hence",
    "similarly", "likewise", "in the same way",
    "specifically", "in particular", "notably", "for example", "for instance",
    "meanwhile", "subsequently", "ultimately", "overall",
    "first", "second", "third", "finally",
]


def _extract_section(report: str, header: str) -> str:
    """Extract text under a markdown header until the next header of equal or higher level."""
    pattern = re.compile(
        rf"^(#{{1,3}})\s+.*{re.escape(header)}.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(report)
    if not match:
        return ""
    level = len(match.group(1))
    start = match.end()
    next_header = re.compile(rf"^#{{{1},{level}}}\s+", re.MULTILINE)
    next_match = next_header.search(report, start)
    end = next_match.start() if next_match else len(report)
    return report[start:end].strip()


def _parse_reference_list(report: str) -> dict[int, dict]:
    """Parse numbered reference entries from the References section.

    Returns {number: {"title": str, "url": str}}.
    """
    refs_section = _extract_section(report, "References")
    refs: dict[int, dict] = {}
    for m in _REFERENCE_LINK_RE.finditer(refs_section):
        num = int(m.group(1))
        refs[num] = {"title": m.group(2), "url": m.group(3)}
    if not refs:
        for m in _MARKDOWN_LINK_RE.finditer(refs_section):
            idx = len(refs) + 1
            refs[idx] = {"title": m.group(1), "url": m.group(2)}
    return refs


# ===========================================================================
# 1. W/ GT & Code -- Objective evaluations against ground truth
# ===========================================================================


@weave.op()
def writer_citation_mapping_accuracy(output: dict, target: dict) -> dict:
    """Verify inline citation numbers match the correct URL/title from ground truth.

    For each expected citation number, checks that the report's reference list
    entry has a URL whose domain matches the ground truth URL's domain.
    """
    report = output.get("draft_report", "")
    expected_citations = target.get("expected_citations", {})
    if not expected_citations:
        return {"citation_mapping_accuracy": 1.0, "citation_mapping_hits": 0, "citation_mapping_total": 0, "details": []}

    parsed_refs = _parse_reference_list(report)
    hits = 0
    details = []
    for num_str, gt in expected_citations.items():
        num = int(num_str)
        gt_domain = urlparse(gt["url"]).netloc.replace("www.", "")
        report_ref = parsed_refs.get(num)
        if report_ref:
            report_domain = urlparse(report_ref["url"]).netloc.replace("www.", "")
            matched = gt_domain in report_domain or report_domain in gt_domain
        else:
            matched = False
        if matched:
            hits += 1
        details.append({
            "citation_number": num,
            "expected_domain": gt_domain,
            "found_domain": urlparse(report_ref["url"]).netloc if report_ref else None,
            "matched": matched,
        })

    total = len(expected_citations)
    return {
        "citation_mapping_accuracy": hits / total if total else 1.0,
        "citation_mapping_hits": hits,
        "citation_mapping_total": total,
        "details": details,
    }


@weave.op()
def writer_key_metric_retention(output: dict, target: dict) -> dict:
    """Ensure specific must-include data points from analysis appear in the report."""
    report = output.get("draft_report", "")
    must_include = target.get("must_include_metrics", [])
    if not must_include:
        return {"metric_retention": 1.0, "metric_hits": 0, "metric_total": 0, "missing_metrics": []}

    hits = 0
    missing = []
    for metric in must_include:
        if metric in report:
            hits += 1
        else:
            missing.append(metric)

    return {
        "metric_retention": hits / len(must_include),
        "metric_hits": hits,
        "metric_total": len(must_include),
        "missing_metrics": missing,
    }


@weave.op()
def writer_reference_link_integrity(output: dict, target: dict) -> dict:
    """Check that all reference links have valid domains matching expected sources.

    Verifies that each numbered reference in the report exists and its domain
    appears in the ground truth expected_reference_domains list.
    """
    report = output.get("draft_report", "")
    expected_domains = target.get("expected_reference_domains", [])
    if not expected_domains:
        return {"reference_integrity": 1.0, "domain_hits": 0, "domain_total": 0, "details": []}

    parsed_refs = _parse_reference_list(report)
    expected_normalized = [d.replace("www.", "").lower() for d in expected_domains]

    domain_hits = 0
    details = []
    for expected_domain in expected_normalized:
        found = False
        for ref in parsed_refs.values():
            ref_domain = urlparse(ref["url"]).netloc.replace("www.", "").lower()
            if expected_domain in ref_domain or ref_domain in expected_domain:
                found = True
                break
        if found:
            domain_hits += 1
        details.append({"expected_domain": expected_domain, "found": found})

    total = len(expected_domains)
    return {
        "reference_integrity": domain_hits / total if total else 1.0,
        "domain_hits": domain_hits,
        "domain_total": total,
        "details": details,
    }


# ===========================================================================
# 2. W/o GT & Code -- Structural and formatting checks (no ground truth)
# ===========================================================================


@weave.op()
def writer_section_completeness(output: dict) -> dict:
    """Verify presence of all 8 mandatory sections."""
    report = output.get("draft_report", "")
    report_lower = report.lower()

    present = []
    missing = []
    for section in REQUIRED_SECTIONS:
        if section.lower() == "title":
            has_title = bool(re.search(r"^#\s+.+", report, re.MULTILINE))
            (present if has_title else missing).append(section)
        else:
            header_pattern = re.compile(
                rf"^#{{1,3}}\s+.*{re.escape(section.lower())}",
                re.MULTILINE | re.IGNORECASE,
            )
            (present if header_pattern.search(report) else missing).append(section)

    return {
        "section_completeness": len(present) / len(REQUIRED_SECTIONS),
        "sections_present": present,
        "sections_missing": missing,
        "total_required": len(REQUIRED_SECTIONS),
    }


@weave.op()
def writer_word_count_verification(output: dict) -> dict:
    """Count words in the Findings section and check against 1500-3000 target."""
    report = output.get("draft_report", "")
    findings_text = _extract_section(report, "Findings")
    word_count = len(findings_text.split()) if findings_text else 0
    in_range = 1500 <= word_count <= 3000
    total_word_count = len(report.split())

    return {
        "findings_word_count": word_count,
        "findings_in_target_range": in_range,
        "total_report_word_count": total_word_count,
    }


@weave.op()
def writer_markdown_syntax_audit(output: dict) -> dict:
    """Check for common Markdown syntax issues."""
    report = output.get("draft_report", "")
    issues = []

    unclosed_bold = re.findall(r"(?<!\*)\*\*(?!\*)([^*]+)$", report, re.MULTILINE)
    if unclosed_bold:
        issues.append(f"Unclosed bold tags: {len(unclosed_bold)} instance(s)")

    unclosed_italic = re.findall(r"(?<!\*)\*(?!\*)([^*]+)$", report, re.MULTILINE)
    unclosed_italic = [m for m in unclosed_italic if not m.startswith("*")]
    if unclosed_italic:
        issues.append(f"Unclosed italic tags: {len(unclosed_italic)} instance(s)")

    lines = report.split("\n")
    prev_level = 0
    nesting_issues = 0
    for line in lines:
        header_match = re.match(r"^(#{1,6})\s+", line)
        if header_match:
            level = len(header_match.group(1))
            if prev_level > 0 and level > prev_level + 1:
                nesting_issues += 1
            prev_level = level
    if nesting_issues:
        issues.append(f"Header nesting skips: {nesting_issues} instance(s)")

    refs_section = _extract_section(report, "References")
    if refs_section:
        ref_lines = [l.strip() for l in refs_section.split("\n") if l.strip()]
        malformed_refs = 0
        for line in ref_lines:
            if re.match(r"^\d+\.", line):
                if not _MARKDOWN_LINK_RE.search(line):
                    malformed_refs += 1
        if malformed_refs:
            issues.append(f"Malformed reference links: {malformed_refs} instance(s)")

    inline_citations = set(_INLINE_CITATION_RE.findall(report))
    parsed_refs = _parse_reference_list(report)
    ref_numbers = {str(n) for n in parsed_refs.keys()}
    dangling = inline_citations - ref_numbers
    if dangling:
        issues.append(f"Dangling citations with no reference entry: {sorted(dangling)}")

    return {
        "markdown_valid": len(issues) == 0,
        "markdown_issues": issues,
        "markdown_issue_count": len(issues),
    }


@weave.op()
def writer_transition_density(output: dict) -> dict:
    """Calculate frequency of transition words to assess narrative flow."""
    report = output.get("draft_report", "")
    report_lower = report.lower()
    total_words = len(report.split())
    if total_words == 0:
        return {"transition_count": 0, "transition_density": 0.0, "transitions_per_1000_words": 0.0}

    transition_count = 0
    found_transitions: dict[str, int] = {}
    for tw in TRANSITION_WORDS:
        count = len(re.findall(rf"\b{re.escape(tw)}\b", report_lower))
        if count > 0:
            transition_count += count
            found_transitions[tw] = count

    density = transition_count / total_words
    per_1000 = (transition_count / total_words) * 1000

    return {
        "transition_count": transition_count,
        "transition_density": round(density, 4),
        "transitions_per_1000_words": round(per_1000, 1),
        "transition_breakdown": found_transitions,
    }


# ===========================================================================
# 3. W/ GT & LLM -- Subjective evaluation against gold standard
# ===========================================================================


class WriterToneAlignmentScorer(Scorer):
    """LLM judge: compare the report's tone against the gold standard report."""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, target: dict, research_query: str) -> dict:
        report = output.get("draft_report", "")
        gold_standard = target.get("gold_standard_report", "")
        if not gold_standard:
            return {"tone_alignment": 1, "explanation": "No gold standard provided."}

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert writing evaluator. Compare the tone of the "
                        "candidate report against the gold standard report.\n\n"
                        "The expected tone is 'academic-adjacent': professional, "
                        "evidence-based, measured, and objective, but accessible to a "
                        "general educated audience (not overly jargon-heavy).\n\n"
                        "Evaluate on a scale from 1-10 where:\n"
                        "- 1-3: Significantly different tone (too casual, too formal, "
                        "or inconsistent)\n"
                        "- 4-6: Partially aligned but noticeable deviations\n"
                        "- 7-9: Well-aligned with minor differences\n"
                        "- 10: Essentially identical tone\n\n"
                        "Return JSON:\n"
                        '{"tone_score": <int 1-10>, "explanation": "<reasoning>"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"--- GOLD STANDARD REPORT ---\n{gold_standard[:3000]}\n\n"
                        f"--- CANDIDATE REPORT ---\n{report[:3000]}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        tone_score = max(1, min(10, int(parsed.get("tone_score", 5))))
        return {
            "tone_alignment": tone_score,
            "explanation": parsed.get("explanation", ""),
        }


class WriterSynthesisFidelityScorer(Scorer):
    """LLM judge: identify nuances from the gold standard that the writer omitted."""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, target: dict, research_query: str) -> dict:
        report = output.get("draft_report", "")
        gold_standard = target.get("gold_standard_report", "")
        if not gold_standard:
            return {
                "synthesis_fidelity": 1.0,
                "lost_insights": [],
                "explanation": "No gold standard provided.",
            }

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert research evaluator. Compare the candidate "
                        "report against the gold standard and identify any crucial "
                        "nuances, insights, or analytical points that are present in "
                        "the gold standard but MISSING from the candidate.\n\n"
                        "Focus on substantive analytical insights, not surface-level "
                        "wording differences. Look for:\n"
                        "- Key arguments or interpretations that were dropped\n"
                        "- Important caveats or qualifications that were omitted\n"
                        "- Connections between data points that were lost\n"
                        "- Critical implications that were not addressed\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "lost_insights": ["<insight 1>", "<insight 2>", ...],\n'
                        '  "total_key_insights_in_gold": <int>,\n'
                        '  "insights_retained": <int>,\n'
                        '  "explanation": "<reasoning>"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"--- GOLD STANDARD REPORT ---\n{gold_standard[:3000]}\n\n"
                        f"--- CANDIDATE REPORT ---\n{report[:3000]}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        lost = parsed.get("lost_insights", [])
        total = max(1, int(parsed.get("total_key_insights_in_gold", 1)))
        retained = int(parsed.get("insights_retained", total - len(lost)))
        fidelity = max(0.0, min(1.0, retained / total))

        return {
            "synthesis_fidelity": round(fidelity, 3),
            "lost_insights": lost,
            "total_key_insights": total,
            "insights_retained": retained,
            "explanation": parsed.get("explanation", ""),
        }


# ===========================================================================
# 4. W/o GT & LLM -- Binary rubric (no ground truth)
# ===========================================================================

_WRITER_RUBRIC_CRITERIA = [
    "executive_summary_quality",
    "thematic_cohesion",
    "citation_consistency",
    "implications_depth",
]


class WriterRubricScorer(Scorer):
    """LLM judge: 4-point binary rubric for professional presentation and logic.

    Criteria (each scored 1 or 0):
    1. Executive Summary Quality -- standalone overview vs too vague
    2. Thematic Cohesion -- Findings organized by logical themes vs data dump
    3. Citation Consistency -- inline citations used consistently across all sections
    4. Implications Depth -- Discussion offers genuine interpretation vs repeating findings
    """

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, research_query: str) -> dict:
        report = output.get("draft_report", "")

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate a research report against a 4-criterion rubric. "
                        "For each criterion return exactly 1 (pass) or 0 (fail).\n\n"
                        "Criteria:\n"
                        "1. executive_summary_quality -- Does the Executive Summary "
                        "provide a standalone overview that a reader could understand "
                        "without reading the rest of the report? Score 0 if it is too "
                        "vague, too short, or merely restates the introduction.\n\n"
                        "2. thematic_cohesion -- Within the Findings section, are "
                        "subsections organized by logical themes (e.g., costs, "
                        "scalability, policy) rather than being a dump of unconnected "
                        "data points? Score 0 if findings lack thematic structure.\n\n"
                        "3. citation_consistency -- Are inline citations [1], [2], etc. "
                        "used consistently across every major section of the report, "
                        "particularly in the Discussion and Findings? Score 0 if "
                        "citations are concentrated in only one section or largely "
                        "absent from Discussion/Findings.\n\n"
                        "4. implications_depth -- In the Discussion section, does the "
                        "writer offer genuine interpretation, analysis, or implications "
                        "of the data? Score 0 if Discussion merely repeats or "
                        "summarizes the Findings without adding new insight.\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "executive_summary_quality": 0 or 1,\n'
                        '  "thematic_cohesion": 0 or 1,\n'
                        '  "citation_consistency": 0 or 1,\n'
                        '  "implications_depth": 0 or 1,\n'
                        '  "explanation": "<brief reasoning for each criterion>"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Report:\n{report[:5000]}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        scores = {c: int(bool(parsed.get(c, 0))) for c in _WRITER_RUBRIC_CRITERIA}
        total = sum(scores.values())

        return {
            **scores,
            "rubric_total": total,
            "rubric_score": total / len(_WRITER_RUBRIC_CRITERIA),
            "explanation": parsed.get("explanation", ""),
        }


# ===========================================================================
# Registry
# ===========================================================================


def get_scorers() -> list:
    return [
        # 1. W/ GT & Code
        writer_citation_mapping_accuracy,
        writer_key_metric_retention,
        writer_reference_link_integrity,
        # 2. W/o GT & Code
        writer_section_completeness,
        writer_word_count_verification,
        writer_markdown_syntax_audit,
        writer_transition_density,
        # 3. W/ GT & LLM
        WriterToneAlignmentScorer(),
        WriterSynthesisFidelityScorer(),
        # 4. W/o GT & LLM
        WriterRubricScorer(),
    ]
