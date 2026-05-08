"""Format checker agent scorers (all 4 quadrants of the 2x2 eval matrix).

1. W/ GT  & Code  -- Sabotage detection rate, citation cross-reference accuracy,
                     path validation consistency
2. W/o GT & Code  -- JSON schema validation, severity logic check, heading hierarchy
                     check, placeholder detection
3. W/ GT  & LLM   -- False positive audit, suggestion usefulness
4. W/o GT & LLM   -- 4-point binary rubric (substantive analysis, technical precision,
                     clarity of summary, markdown knowledge)
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

_INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+", re.MULTILINE)
_PLACEHOLDER_RE = re.compile(
    r"\bTODO\b|\bTBD\b|\[Insert\s[^\]]*\]|\bFIXME\b|\bXXX\b",
    re.IGNORECASE,
)

_REQUIRED_SECTIONS = [
    "Executive Summary",
    "Introduction",
    "Methodology",
    "Findings",
    "Discussion",
    "Conclusion",
    "References",
]


def _issues_mention_any(issues: list[dict], keywords: list[str]) -> list[dict]:
    """Return issues whose description or section matches any keyword (case-insensitive)."""
    matched = []
    for iss in issues:
        text = " ".join([
            str(iss.get("description", "")),
            str(iss.get("section", "")),
            str(iss.get("suggestion", "")),
        ]).lower()
        if any(kw.lower() in text for kw in keywords):
            matched.append(iss)
    return matched


def _parse_reference_list(report: str) -> dict[int, str]:
    """Return {ref_number: url_or_title} from the References section."""
    refs_section = ""
    match = re.search(
        r"^#{1,3}\s+References\s*$",
        report,
        re.MULTILINE | re.IGNORECASE,
    )
    if match:
        refs_section = report[match.end():]
    entries: dict[int, str] = {}
    for m in re.finditer(r"^\s*(\d+)\.\s*\[([^\]]*)\]\(([^)]+)\)", refs_section, re.MULTILINE):
        entries[int(m.group(1))] = m.group(3)
    return entries


# ===========================================================================
# 1. W/ GT & Code -- Objective evaluations against ground truth
# ===========================================================================


@weave.op()
def format_checker_sabotage_detection(output: dict, target: dict) -> dict:
    """Measure ratio of planted errors that the agent successfully detected.

    Each planted error has a `type` and `detail`. We check whether the agent's
    issues list contains at least one entry whose description/section plausibly
    references the planted error.
    """
    planted = target.get("planted_errors", [])
    if not planted:
        return {
            "sabotage_detected": 0,
            "sabotage_total": 0,
            "sabotage_detection_rate": 1.0,
            "details": [],
        }

    checker = output.get("checker_result", {})
    agent_issues = checker.get("issues", [])

    keyword_map = {
        "missing_section": ["methodology", "missing", "required section"],
        "dangling_citation": ["citation", "[4]", "reference", "dangling"],
        "orphan_reference": ["reference", "cited", "unused", "orphan", "never cited", "not cited"],
        "invalid_image_path": ["image", "path", "chart", "nonexistent", "invalid"],
        "placeholder_text": ["todo", "placeholder", "tbd", "insert"],
        "heading_hierarchy_skip": ["heading", "hierarchy", "skip", "h2", "h4", "level"],
    }

    detected = 0
    details = []
    for error in planted:
        error_type = error.get("type", "")
        search_terms = keyword_map.get(error_type, [error_type])
        matches = _issues_mention_any(agent_issues, search_terms)
        found = len(matches) > 0
        if found:
            detected += 1
        details.append({
            "planted_error": error["detail"],
            "type": error_type,
            "detected": found,
            "matching_issues": len(matches),
        })

    return {
        "sabotage_detected": detected,
        "sabotage_total": len(planted),
        "sabotage_detection_rate": detected / len(planted),
        "details": details,
    }


@weave.op()
def format_checker_citation_cross_ref(output: dict, target: dict, draft_report: str) -> dict:
    """Verify the agent correctly identifies citation/reference mismatches.

    Checks two directions:
    - Citations used in text but missing from the reference list
    - References in the list that are never cited in the text
    """
    expected = target.get("expected_citation_mismatches", {})
    if not expected:
        return {
            "citation_crossref_accuracy": 1.0,
            "dangling_hits": 0,
            "dangling_total": 0,
            "orphan_hits": 0,
            "orphan_total": 0,
        }

    checker = output.get("checker_result", {})
    agent_issues = checker.get("issues", [])
    issues_text = " ".join(
        str(iss.get("description", "")) + " " + str(iss.get("suggestion", ""))
        for iss in agent_issues
    ).lower()

    expected_dangling = expected.get("cited_but_not_in_references", [])
    expected_orphans = expected.get("in_references_but_not_cited", [])

    dangling_hits = 0
    for cite_num in expected_dangling:
        if f"[{cite_num}]" in issues_text or f"citation {cite_num}" in issues_text:
            dangling_hits += 1

    orphan_hits = 0
    for ref_num in expected_orphans:
        if f"[{ref_num}]" in issues_text or f"reference {ref_num}" in issues_text or f"#{ref_num}" in issues_text:
            orphan_hits += 1

    total_checks = len(expected_dangling) + len(expected_orphans)
    total_hits = dangling_hits + orphan_hits

    return {
        "citation_crossref_accuracy": total_hits / total_checks if total_checks else 1.0,
        "dangling_hits": dangling_hits,
        "dangling_total": len(expected_dangling),
        "orphan_hits": orphan_hits,
        "orphan_total": len(expected_orphans),
    }


@weave.op()
def format_checker_path_validation(output: dict, target: dict) -> dict:
    """Check if the agent correctly flagged invalid image paths and accepted valid ones."""
    invalid_paths = target.get("invalid_image_paths", [])
    valid_paths = target.get("valid_image_paths", [])

    if not invalid_paths and not valid_paths:
        return {
            "path_validation_accuracy": 1.0,
            "invalid_flagged": 0,
            "invalid_total": 0,
            "false_flags_on_valid": 0,
            "valid_total": 0,
        }

    checker = output.get("checker_result", {})
    agent_issues = checker.get("issues", [])
    issues_text = " ".join(
        str(iss.get("description", "")) + " " + str(iss.get("suggestion", ""))
        for iss in agent_issues
    ).lower()

    invalid_flagged = 0
    for path in invalid_paths:
        path_stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if path.lower() in issues_text or path_stem in issues_text:
            invalid_flagged += 1

    false_flags = 0
    for path in valid_paths:
        path_stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if path.lower() in issues_text or path_stem in issues_text:
            false_flags += 1

    correct = invalid_flagged + (len(valid_paths) - false_flags)
    total = len(invalid_paths) + len(valid_paths)

    return {
        "path_validation_accuracy": correct / total if total else 1.0,
        "invalid_flagged": invalid_flagged,
        "invalid_total": len(invalid_paths),
        "false_flags_on_valid": false_flags,
        "valid_total": len(valid_paths),
    }


# ===========================================================================
# 2. W/o GT & Code -- Structural checks (no ground truth needed)
# ===========================================================================


@weave.op()
def format_checker_json_schema(output: dict) -> dict:
    """Validate that the checker output strictly follows the expected JSON schema.

    Expected: checker_result contains `passed` (bool), `issues` (list of objects
    each with severity, section, description, suggestion), and `summary` (str).
    """
    checker = output.get("checker_result", {})

    has_passed = isinstance(checker.get("passed"), bool)
    has_summary = isinstance(checker.get("summary"), str)

    issues = checker.get("issues", [])
    has_issues_list = isinstance(issues, list)

    valid_issues = True
    issue_field_errors = []
    required_fields = {"severity", "section", "description", "suggestion"}
    for i, iss in enumerate(issues):
        if not isinstance(iss, dict):
            valid_issues = False
            issue_field_errors.append(f"Issue {i} is not a dict")
            continue
        missing = required_fields - set(iss.keys())
        if missing:
            valid_issues = False
            issue_field_errors.append(f"Issue {i} missing fields: {sorted(missing)}")
        if "severity" in iss and iss["severity"] not in ("error", "warning"):
            valid_issues = False
            issue_field_errors.append(
                f"Issue {i} has invalid severity: {iss['severity']}"
            )

    schema_valid = all([has_passed, has_summary, has_issues_list, valid_issues])

    return {
        "schema_valid": schema_valid,
        "has_passed_field": has_passed,
        "has_summary_field": has_summary,
        "has_issues_list": has_issues_list,
        "all_issues_well_formed": valid_issues,
        "issue_field_errors": issue_field_errors,
    }


@weave.op()
def format_checker_severity_logic(output: dict) -> dict:
    """If any issue has severity 'error', the `passed` field must be false."""
    checker = output.get("checker_result", {})
    passed = checker.get("passed", True)
    issues = checker.get("issues", [])

    has_errors = any(
        isinstance(iss, dict) and iss.get("severity") == "error"
        for iss in issues
    )

    logic_consistent = True
    if has_errors and passed:
        logic_consistent = False

    return {
        "severity_logic_consistent": logic_consistent,
        "has_error_severity": has_errors,
        "passed_value": passed,
    }


@weave.op()
def format_checker_heading_hierarchy(output: dict, draft_report: str) -> dict:
    """Verify the agent detects heading hierarchy skips in the input report.

    Independently checks the input for h-level jumps (e.g., h2 followed by h4)
    and then verifies the agent flagged them.
    """
    headings = _HEADING_PATTERN.findall(draft_report)
    levels = [len(h) for h in headings]

    actual_skips = []
    for i in range(1, len(levels)):
        if levels[i] > levels[i - 1] + 1:
            actual_skips.append({
                "from_level": levels[i - 1],
                "to_level": levels[i],
                "position": i,
            })

    if not actual_skips:
        return {
            "heading_skips_in_input": 0,
            "heading_check_passed": True,
        }

    checker = output.get("checker_result", {})
    agent_issues = checker.get("issues", [])
    heading_issues = _issues_mention_any(
        agent_issues,
        ["heading", "hierarchy", "skip", "level", "h1", "h2", "h3", "h4", "h5", "h6"],
    )

    detected = len(heading_issues) > 0

    return {
        "heading_skips_in_input": len(actual_skips),
        "heading_skips_detected": detected,
        "heading_check_passed": detected,
        "skip_details": actual_skips,
    }


@weave.op()
def format_checker_placeholder_detection(output: dict, draft_report: str) -> dict:
    """Run regex on the input to find placeholders, then verify the agent flagged them."""
    actual_placeholders = _PLACEHOLDER_RE.findall(draft_report)

    if not actual_placeholders:
        return {
            "placeholders_in_input": 0,
            "placeholder_check_passed": True,
        }

    checker = output.get("checker_result", {})
    agent_issues = checker.get("issues", [])
    placeholder_issues = _issues_mention_any(
        agent_issues,
        ["placeholder", "todo", "tbd", "insert", "incomplete"],
    )

    detected = len(placeholder_issues) > 0

    return {
        "placeholders_in_input": len(actual_placeholders),
        "placeholders_detected": detected,
        "placeholder_check_passed": detected,
        "actual_placeholders": actual_placeholders,
    }


# ===========================================================================
# 3. W/ GT & LLM -- Subjective evaluation against expert ground truth
# ===========================================================================


class FormatCheckerFalsePositiveScorer(Scorer):
    """LLM judge: compare the agent's issues against an expert report to find
    false positives -- things the agent flagged that are actually correct."""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, target: dict, draft_report: str) -> dict:
        expert_issues = target.get("expert_issues", [])
        checker = output.get("checker_result", {})
        agent_issues = checker.get("issues", [])

        if not agent_issues:
            return {
                "false_positive_count": 0,
                "total_agent_issues": 0,
                "false_positive_rate": 0.0,
                "explanation": "Agent reported no issues.",
            }

        agent_text = "\n".join(
            f"- [{iss.get('severity', '?')}] {iss.get('section', '?')}: "
            f"{iss.get('description', '?')}"
            for iss in agent_issues
            if isinstance(iss, dict)
        )
        expert_text = "\n".join(
            f"- [{iss.get('severity', '?')}] {iss.get('section', '?')}: "
            f"{iss.get('description', '?')}"
            for iss in expert_issues
            if isinstance(iss, dict)
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Markdown and report formatting evaluator. "
                        "You will be shown a Markdown report, the issues an automated "
                        "checker flagged, and an expert's list of real issues.\n\n"
                        "Your task: identify FALSE POSITIVES in the checker's list -- "
                        "issues the checker flagged that are NOT actual problems in the "
                        "report. An issue is a false positive if the report's Markdown "
                        "is actually correct in that regard.\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "false_positives": [\n'
                        '    {"checker_issue": "<description>", "reason": "<why it is not a real issue>"}\n'
                        "  ],\n"
                        '  "explanation": "<overall assessment>"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"--- REPORT ---\n{draft_report[:3000]}\n\n"
                        f"--- CHECKER'S ISSUES ---\n{agent_text}\n\n"
                        f"--- EXPERT'S REAL ISSUES ---\n{expert_text}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        fps = parsed.get("false_positives", [])

        total = len(agent_issues)
        fp_rate = len(fps) / total if total else 0.0

        return {
            "false_positive_count": len(fps),
            "total_agent_issues": total,
            "false_positive_rate": round(fp_rate, 3),
            "false_positives": fps,
            "explanation": parsed.get("explanation", ""),
        }


class FormatCheckerSuggestionScorer(Scorer):
    """LLM judge: grade the `suggestion` field in each issue for technical
    correctness and implementability."""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, target: dict, draft_report: str) -> dict:
        checker = output.get("checker_result", {})
        agent_issues = checker.get("issues", [])

        issues_with_suggestions = [
            iss for iss in agent_issues
            if isinstance(iss, dict) and iss.get("suggestion", "").strip()
        ]

        if not issues_with_suggestions:
            return {
                "suggestion_quality": 1.0,
                "suggestions_graded": 0,
                "explanation": "No suggestions to evaluate.",
            }

        issues_text = "\n".join(
            f"{i+1}. Issue: {iss.get('description', '?')}\n"
            f"   Suggestion: {iss.get('suggestion', '?')}"
            for i, iss in enumerate(issues_with_suggestions)
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate the quality of fix suggestions provided by a "
                        "Markdown format checker. For each suggestion, determine:\n"
                        "1. Is the suggested fix technically correct?\n"
                        "2. Is it specific enough for a human writer to implement?\n\n"
                        "Grade each suggestion as 1 (useful) or 0 (not useful).\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "grades": {"1": 0 or 1, "2": 0 or 1, ...},\n'
                        '  "explanation": "<reasoning for each grade>"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"--- ORIGINAL REPORT ---\n{draft_report[:2000]}\n\n"
                        f"--- ISSUES AND SUGGESTIONS ---\n{issues_text}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        grades = parsed.get("grades", {})

        total = len(issues_with_suggestions)
        hits = sum(int(bool(grades.get(str(i + 1), 0))) for i in range(total))

        return {
            "suggestion_quality": hits / total if total else 1.0,
            "suggestion_hits": hits,
            "suggestions_graded": total,
            "explanation": parsed.get("explanation", ""),
        }


# ===========================================================================
# 4. W/o GT & LLM -- Linter rigor binary rubric (no ground truth)
# ===========================================================================

_LINTER_RUBRIC_CRITERIA = [
    "substantive_analysis",
    "technical_precision",
    "clarity_of_summary",
    "markdown_knowledge",
]


class FormatCheckerLinterRigorScorer(Scorer):
    """LLM judge: 4-point binary rubric evaluating the checker's rigor.

    Criteria (each scored 1 or 0):
    1. Substantive Analysis -- Did the agent distinguish between a section that is
       "short" vs. one that is "empty" or just headers?
    2. Technical Precision -- Does the agent specify exact citation numbers or exact
       lines that are broken, rather than vague "citations are wrong" messages?
    3. Clarity of Summary -- Does the summary provide a high-level executive view
       of the report's health?
    4. Markdown Knowledge -- Does the agent correctly identify issues with complex
       Markdown elements like tables or nested lists?
    """

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, draft_report: str) -> dict:
        checker = output.get("checker_result", {})
        agent_issues = checker.get("issues", [])
        summary = checker.get("summary", "")

        issues_text = "\n".join(
            f"- [{iss.get('severity', '?')}] {iss.get('section', '?')}: "
            f"{iss.get('description', '?')} | Suggestion: {iss.get('suggestion', 'N/A')}"
            for iss in agent_issues
            if isinstance(iss, dict)
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate a Markdown format checker's output against a "
                        "4-criterion rigor rubric. For each criterion return exactly "
                        "1 (pass) or 0 (fail).\n\n"
                        "Criteria:\n"
                        "1. substantive_analysis -- Did the checker distinguish between "
                        "a section that is merely 'short' versus one that is 'empty' "
                        "or contains only headers with no body text? Score 1 if the "
                        "checker's issues demonstrate this nuanced distinction, or if "
                        "no such ambiguity exists. Score 0 if the checker conflates "
                        "short content with missing content.\n\n"
                        "2. technical_precision -- Does the checker specify exact "
                        "citation numbers (e.g., '[4]'), exact section names, or exact "
                        "elements that are broken? Score 1 if issues are specific. "
                        "Score 0 if the checker gives vague statements like 'citations "
                        "are wrong' without specifying which ones.\n\n"
                        "3. clarity_of_summary -- Does the summary field provide a "
                        "high-level executive overview of the report's formatting "
                        "health (e.g., 'Structurally sound but lacks proper citation "
                        "mapping')? Score 1 if the summary is informative and concise. "
                        "Score 0 if it is absent, generic, or just repeats the issues.\n\n"
                        "4. markdown_knowledge -- Does the checker demonstrate "
                        "understanding of Markdown syntax by correctly identifying "
                        "issues with headings, links, images, lists, or other Markdown "
                        "constructs? Score 1 if the checker shows Markdown-aware "
                        "analysis. Score 0 if it treats the report as plain text.\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "substantive_analysis": 0 or 1,\n'
                        '  "technical_precision": 0 or 1,\n'
                        '  "clarity_of_summary": 0 or 1,\n'
                        '  "markdown_knowledge": 0 or 1,\n'
                        '  "explanation": "<brief reasoning for each criterion>"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"--- ORIGINAL REPORT ---\n{draft_report[:3000]}\n\n"
                        f"--- CHECKER'S ISSUES ---\n{issues_text}\n\n"
                        f"--- CHECKER'S SUMMARY ---\n{summary}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        scores = {c: int(bool(parsed.get(c, 0))) for c in _LINTER_RUBRIC_CRITERIA}
        total = sum(scores.values())

        return {
            **scores,
            "rubric_total": total,
            "rubric_score": total / len(_LINTER_RUBRIC_CRITERIA),
            "explanation": parsed.get("explanation", ""),
        }


# ===========================================================================
# Registry
# ===========================================================================


def get_scorers() -> list:
    return [
        # 1. W/ GT & Code
        format_checker_sabotage_detection,
        format_checker_citation_cross_ref,
        format_checker_path_validation,
        # 2. W/o GT & Code
        format_checker_json_schema,
        format_checker_severity_logic,
        format_checker_heading_hierarchy,
        format_checker_placeholder_detection,
        # 3. W/ GT & LLM
        FormatCheckerFalsePositiveScorer(),
        FormatCheckerSuggestionScorer(),
        # 4. W/o GT & LLM
        FormatCheckerLinterRigorScorer(),
    ]
