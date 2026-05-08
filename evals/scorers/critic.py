"""Critic agent scorers (all 4 quadrants of the 2x2 eval matrix + loop meta-eval).

1. W/ GT  & Code  – Verdict accuracy (existing)
2. W/o GT & Code  – JSON schema/enums, boolean consistency, query generation
3. W/ GT  & LLM   – Feedback alignment, severity calibration
4. W/o GT & LLM   – 4-point binary rubric (actionability, bias detection, query quality, logical depth)
5. Loop meta-eval – Convergence rate, resolution quality
"""

from __future__ import annotations

import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

_ALLOWED_QUALITY_VALUES = {"excellent", "good", "fair", "poor"}


# ---------------------------------------------------------------------------
# 1. W/ GT & Code – Verdict accuracy
# ---------------------------------------------------------------------------


@weave.op()
def critic_verdict_accuracy(output: dict, target: dict) -> dict:
    """Check whether the critic's approval matches the expected verdict."""
    critique = output.get("critique", {})
    expected_approved = target.get("expected_approved")
    if expected_approved is None:
        return {"verdict_correct": True}
    return {"verdict_correct": critique.get("approved") == expected_approved}


# ---------------------------------------------------------------------------
# 2. W/o GT & Code – JSON schema/enums, boolean consistency, query generation
# ---------------------------------------------------------------------------


@weave.op()
def critic_schema_and_enums(output: dict) -> dict:
    """Verify all required keys exist and overall_quality uses allowed values."""
    critique = output.get("critique", {})
    required_keys = {"approved", "overall_quality", "gaps", "feedback", "additional_search_queries"}
    present_keys = set(critique.keys()) & required_keys
    all_keys_present = present_keys == required_keys

    quality = critique.get("overall_quality", "")
    quality_valid = quality in _ALLOWED_QUALITY_VALUES

    approved_is_bool = isinstance(critique.get("approved"), bool)
    gaps_is_list = isinstance(critique.get("gaps"), list)
    queries_is_list = isinstance(critique.get("additional_search_queries"), list)
    feedback_is_string = isinstance(critique.get("feedback"), str)

    checks = [all_keys_present, quality_valid, approved_is_bool, gaps_is_list, queries_is_list, feedback_is_string]
    return {
        "all_keys_present": all_keys_present,
        "missing_keys": list(required_keys - present_keys),
        "quality_valid": quality_valid,
        "quality_value": quality,
        "approved_is_bool": approved_is_bool,
        "gaps_is_list": gaps_is_list,
        "queries_is_list": queries_is_list,
        "feedback_is_string": feedback_is_string,
        "schema_score": sum(checks) / len(checks),
    }


@weave.op()
def critic_boolean_consistency(output: dict) -> dict:
    """Logic check on boolean field relationships.

    - If approved is True, gaps should be empty or very short (<=1).
    - If the feedback mentions needing more data/analysis, additional_search_queries
      should be non-empty.
    """
    critique = output.get("critique", {})
    approved = critique.get("approved", False)
    gaps = critique.get("gaps", [])
    queries = critique.get("additional_search_queries", [])
    feedback = critique.get("feedback", "")

    if approved:
        approval_gaps_consistent = len(gaps) <= 1
    else:
        approval_gaps_consistent = True

    rejection_has_queries = True
    if not approved:
        rejection_has_queries = len(queries) > 0

    has_feedback = bool(feedback.strip())

    return {
        "approval_gaps_consistent": approval_gaps_consistent,
        "rejection_has_queries": rejection_has_queries,
        "has_feedback": has_feedback,
        "consistency_pass": approval_gaps_consistent and rejection_has_queries and has_feedback,
    }


@weave.op()
def critic_query_generation(output: dict) -> dict:
    """If approved is False, additional_search_queries length must be > 0."""
    critique = output.get("critique", {})
    approved = critique.get("approved", False)
    queries = critique.get("additional_search_queries", [])

    if approved:
        return {
            "query_check_applicable": False,
            "query_count": len(queries),
            "query_generation_pass": True,
        }

    has_queries = len(queries) > 0
    return {
        "query_check_applicable": True,
        "query_count": len(queries),
        "query_generation_pass": has_queries,
    }


# ---------------------------------------------------------------------------
# 3. W/ GT & LLM – Feedback alignment, severity calibration
# ---------------------------------------------------------------------------


class CriticFeedbackAlignmentScorer(Scorer):
    """LLM judge: does the Critic identify the same core weaknesses as the Expert?

    Returns an alignment percentage (0-100).
    """

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(
        self, output: dict, target: dict, research_query: str, analysis: str
    ) -> dict:
        critique = output.get("critique", {})
        feedback = critique.get("feedback", "")
        gaps = critique.get("gaps", [])
        expert_critique = target.get("expert_critique", "")
        if not expert_critique:
            return {"alignment_score": None, "explanation": "No expert critique provided."}

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You compare a Critic agent's feedback against an Expert's "
                        "critique of the same research analysis.\n\n"
                        "Determine what percentage of the Expert's core weaknesses and "
                        "observations are also identified (even if worded differently) "
                        "by the Critic.\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "alignment_score": <int 0-100>,\n'
                        '  "expert_points_found": [<list of expert points the critic caught>],\n'
                        '  "expert_points_missed": [<list of expert points the critic missed>],\n'
                        '  "explanation": "<brief reasoning>"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Analysis excerpt: {analysis[:1500]}\n\n"
                        f"--- CRITIC OUTPUT ---\n"
                        f"Feedback: {feedback}\n"
                        f"Gaps identified: {gaps}\n\n"
                        f"--- EXPERT CRITIQUE ---\n"
                        f"{expert_critique}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        return {
            "alignment_score": parsed.get("alignment_score", 0),
            "expert_points_found": parsed.get("expert_points_found", []),
            "expert_points_missed": parsed.get("expert_points_missed", []),
            "explanation": parsed.get("explanation", ""),
        }


class CriticSeverityCalibrationScorer(Scorer):
    """LLM judge: does the Critic's overall_quality match the Expert's judgment?

    Detects if the Critic is too lenient or too harsh relative to the expert.
    """

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(
        self, output: dict, target: dict, research_query: str, analysis: str
    ) -> dict:
        critique = output.get("critique", {})
        critic_quality = critique.get("overall_quality", "")
        critic_approved = critique.get("approved", False)
        expected_quality = target.get("expected_overall_quality", "")
        if not expected_quality:
            return {"calibration": None, "explanation": "No expected quality provided."}

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate the severity calibration of a Critic agent.\n\n"
                        "The Critic rated the analysis as overall_quality = "
                        f'"{critic_quality}" and approved = {critic_approved}.\n'
                        f'The Expert judged the correct quality as "{expected_quality}".\n\n'
                        "Quality ordering (worst to best): poor < fair < good < excellent.\n\n"
                        "Determine if the Critic is:\n"
                        "- 'calibrated': same quality level as expert (or within 1 step)\n"
                        "- 'lenient': rated higher quality than the expert\n"
                        "- 'harsh': rated lower quality than the expert\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "calibration": "calibrated" | "lenient" | "harsh",\n'
                        '  "critic_quality": "<what critic said>",\n'
                        '  "expected_quality": "<expert judgment>",\n'
                        '  "severity_match": true/false,\n'
                        '  "explanation": "<brief reasoning>"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Analysis excerpt: {analysis[:1000]}\n\n"
                        f"Critic feedback: {critique.get('feedback', '')[:500]}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        return {
            "calibration": parsed.get("calibration", "unknown"),
            "severity_match": parsed.get("severity_match", False),
            "critic_quality": critic_quality,
            "expected_quality": expected_quality,
            "explanation": parsed.get("explanation", ""),
        }


# ---------------------------------------------------------------------------
# 4. W/o GT & LLM – 4-point binary rubric
# ---------------------------------------------------------------------------

_RUBRIC_CRITERIA = [
    "actionability",
    "bias_detection",
    "query_quality",
    "logical_depth",
]


class CriticRubricScorer(Scorer):
    """LLM judge: 4-point binary rubric (no ground truth needed).

    1. Actionability  – Is the feedback specific enough to rewrite the paper?
    2. Bias Detection  – Does the critic comment on balance of analysis?
    3. Query Quality   – Are additional_search_queries keyword-rich (not full sentences)?
    4. Logical Depth   – Does the feedback address reasoning/logic, not just formatting?
    """

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, research_query: str, analysis: str) -> dict:
        critique = output.get("critique", {})
        feedback = critique.get("feedback", "")
        gaps = critique.get("gaps", [])
        queries = critique.get("additional_search_queries", [])

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate a research Critic's output against a 4-criterion "
                        "binary rubric. For each criterion return exactly 1 (pass) or "
                        "0 (fail).\n\n"
                        "Criteria:\n"
                        "1. actionability – The feedback is specific enough that a "
                        "researcher could use it to concretely improve the analysis. "
                        "It names specific missing topics, weak arguments, or missing "
                        'sources. Fail if vague (e.g. "make it better", "needs more '
                        'detail").\n'
                        "2. bias_detection – The critic comments on the *balance* of "
                        "the analysis: whether it presents multiple perspectives, "
                        "addresses counterarguments, or notes one-sidedness. Fail if "
                        "the critic only evaluates content accuracy without mentioning "
                        "perspective balance.\n"
                        "3. query_quality – The additional_search_queries are effective "
                        "keyword-rich search queries (e.g. 'AI labor displacement "
                        "statistics 2024') rather than full natural-language sentences "
                        "(e.g. 'What are the effects of AI on jobs?'). If no queries "
                        "are provided and the analysis is approved, pass. If no queries "
                        "are provided and the analysis is NOT approved, fail.\n"
                        "4. logical_depth – The feedback addresses reasoning quality: "
                        "logical fallacies, unsupported causal claims, missing logical "
                        "steps, or gaps in argumentation. Fail if the feedback only "
                        "addresses formatting, grammar, or surface-level issues.\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "actionability": 0 or 1,\n'
                        '  "bias_detection": 0 or 1,\n'
                        '  "query_quality": 0 or 1,\n'
                        '  "logical_depth": 0 or 1,\n'
                        '  "explanation": "<brief reasoning for each criterion>"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Analysis excerpt: {analysis[:1500]}\n\n"
                        f"--- CRITIC OUTPUT ---\n"
                        f"Approved: {critique.get('approved')}\n"
                        f"Overall quality: {critique.get('overall_quality')}\n"
                        f"Feedback: {feedback}\n"
                        f"Gaps: {gaps}\n"
                        f"Additional search queries: {queries}"
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
# 5. Loop Meta-Eval – Convergence rate & resolution quality
# ---------------------------------------------------------------------------


class CriticLoopEvalScorer(Scorer):
    """Meta-evaluation: measures the Critic's behavior across a revision loop.

    Runs the full Critic -> Analyst -> Critic loop and measures:
    - Convergence rate: how many iterations until approved=True
    - Resolution quality: whether the final analysis improves over the first draft
    """

    model_id: str = "gpt-4o-mini"
    max_loop_iterations: int = 3

    @weave.op()
    def score(self, output: dict, research_query: str, analysis: str) -> dict:
        from agents.analyst import AnalystAgent
        from agents.critic import CriticAgent

        critique = output.get("critique", {})
        if critique.get("approved", False):
            return {
                "convergence_iterations": 0,
                "converged": True,
                "first_draft_quality": critique.get("overall_quality", ""),
                "final_draft_quality": critique.get("overall_quality", ""),
                "quality_improved": False,
                "explanation": "First draft was approved; no loop needed.",
            }

        quality_order = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}
        first_quality = critique.get("overall_quality", "")
        current_analysis = analysis
        current_feedback = critique.get("feedback", "")
        current_queries = critique.get("additional_search_queries", [])

        analyst = AnalystAgent()
        critic = CriticAgent()

        iterations = 0
        converged = False
        final_quality = first_quality

        for i in range(self.max_loop_iterations):
            iterations += 1

            revised_result = analyst.predict(
                research_query=f"{research_query}\n\nRevision feedback: {current_feedback}",
                search_results=[
                    {"title": q, "url": "", "query": q, "content": ""}
                    for q in current_queries
                ] if current_queries else [],
            )
            current_analysis = revised_result.get("analysis", "")

            critic_result = critic.predict(
                research_query=research_query,
                analysis=current_analysis,
                revision_count=i + 1,
                max_revisions=self.max_loop_iterations,
            )
            new_critique = critic_result.get("critique", {})
            final_quality = new_critique.get("overall_quality", "")

            if new_critique.get("approved", False):
                converged = True
                break

            current_feedback = new_critique.get("feedback", "")
            current_queries = new_critique.get("additional_search_queries", [])

        first_rank = quality_order.get(first_quality, -1)
        final_rank = quality_order.get(final_quality, -1)
        quality_improved = final_rank > first_rank

        explanation = self._judge_resolution(
            research_query, analysis, current_analysis, first_quality, final_quality
        )

        return {
            "convergence_iterations": iterations,
            "converged": converged,
            "first_draft_quality": first_quality,
            "final_draft_quality": final_quality,
            "quality_improved": quality_improved,
            "explanation": explanation,
        }

    def _judge_resolution(
        self,
        research_query: str,
        first_analysis: str,
        final_analysis: str,
        first_quality: str,
        final_quality: str,
    ) -> str:
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Compare a first-draft and final-draft research analysis. "
                        "Determine whether the final draft is meaningfully improved.\n\n"
                        "Return JSON:\n"
                        '{"improved": true/false, "explanation": "<brief comparison>"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"FIRST DRAFT (quality: {first_quality}):\n"
                        f"{first_analysis[:1500]}\n\n"
                        f"FINAL DRAFT (quality: {final_quality}):\n"
                        f"{final_analysis[:1500]}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        return parsed.get("explanation", "")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_scorers() -> list:
    return [
        # 1. W/ GT  & Code
        critic_verdict_accuracy,
        # 2. W/o GT & Code
        critic_schema_and_enums,
        critic_boolean_consistency,
        critic_query_generation,
        # 3. W/ GT  & LLM
        CriticFeedbackAlignmentScorer(),
        CriticSeverityCalibrationScorer(),
        # 4. W/o GT & LLM
        CriticRubricScorer(),
        # 5. Loop Meta-Eval
        CriticLoopEvalScorer(),
    ]
