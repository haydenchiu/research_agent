"""Format checker agent scorers."""

from __future__ import annotations

import re

import weave

# ---------------------------------------------------------------------------
# Code / No Ground Truth
# ---------------------------------------------------------------------------

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+", re.MULTILINE)


@weave.op()
def format_checker_structure_check(output: dict) -> dict:
    """Check that the format checker output has valid structure."""
    final_report = output.get("final_report", "")
    format_issues = output.get("format_issues", [])

    has_decision = bool(final_report) or len(format_issues) > 0
    passed = bool(final_report) and len(format_issues) == 0

    valid_heading_hierarchy = True
    if final_report:
        headings = HEADING_PATTERN.findall(final_report)
        levels = [len(h) for h in headings]
        for i in range(1, len(levels)):
            if levels[i] > levels[i - 1] + 1:
                valid_heading_hierarchy = False
                break

    return {
        "has_decision": has_decision,
        "passed": passed,
        "valid_heading_hierarchy": valid_heading_hierarchy,
    }


# ---------------------------------------------------------------------------
# Code / With Ground Truth
# ---------------------------------------------------------------------------


@weave.op()
def format_checker_verdict_accuracy(output: dict, target: dict) -> dict:
    """Check whether the format checker's pass/fail matches the expected result."""
    final_report = output.get("final_report", "")
    format_issues = output.get("format_issues", [])
    actual_passed = bool(final_report) and len(format_issues) == 0

    expected_passed = target.get("expected_passed")
    if expected_passed is None:
        return {"verdict_correct": True}
    return {"verdict_correct": actual_passed == expected_passed}


def get_scorers() -> list:
    return [
        format_checker_structure_check,
        format_checker_verdict_accuracy,
    ]
