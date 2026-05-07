"""Planner agent scorers (all 4 quadrants of the 2x2 eval matrix).

1. W/ GT  & Code  – Semantic similarity matching (embeddings + cosine)
2. W/o GT & Code  – Structural validation (5 checks)
3. W/ GT  & LLM   – Gold-standard concept coverage
4. W/o GT & LLM   – 6-point binary rubric
"""

from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np
import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBED_MODEL = "text-embedding-3-small"


def _get_embeddings(texts: list[str], model: str = _EMBED_MODEL) -> list[list[float]]:
    """Embed a batch of texts via the OpenAI embeddings API."""
    client = OpenAI()
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _extract_question_text(item) -> str:
    """Pull question text from either a dict or a plain string."""
    if isinstance(item, dict):
        return item.get("question", "")
    return str(item)


# ---------------------------------------------------------------------------
# 1. W/ GT & Code – Semantic similarity matching
# ---------------------------------------------------------------------------


@weave.op()
def planner_semantic_similarity(output: dict, target: dict) -> dict:
    """Compare generated questions to GT using embeddings.

    For each GT question, find the best cosine match among generated questions.
    A match counts when similarity >= 0.8.
    """
    generated = output.get("sub_questions", [])
    expected = target.get("expected_sub_questions", [])
    if not expected or not generated:
        return {
            "semantic_matches": 0,
            "semantic_total": len(expected),
            "semantic_recall": 0.0,
            "match_details": [],
        }

    gen_texts = [_extract_question_text(q) for q in generated]
    exp_texts = [_extract_question_text(q) for q in expected]

    all_embeddings = _get_embeddings(gen_texts + exp_texts)
    gen_embs = all_embeddings[: len(gen_texts)]
    exp_embs = all_embeddings[len(gen_texts) :]

    threshold = 0.8
    matches = 0
    details = []
    for i, exp_emb in enumerate(exp_embs):
        best_sim, best_gen = 0.0, ""
        for j, gen_emb in enumerate(gen_embs):
            sim = _cosine_sim(exp_emb, gen_emb)
            if sim > best_sim:
                best_sim = sim
                best_gen = gen_texts[j]
        hit = best_sim >= threshold
        if hit:
            matches += 1
        details.append(
            {
                "expected": exp_texts[i],
                "best_match": best_gen,
                "similarity": round(best_sim, 3),
                "hit": hit,
            }
        )

    return {
        "semantic_matches": matches,
        "semantic_total": len(expected),
        "semantic_recall": matches / len(expected),
        "match_details": details,
    }


# ---------------------------------------------------------------------------
# 2. W/o GT & Code – Structural validation (5 checks)
# ---------------------------------------------------------------------------

_DUPLICATE_THRESHOLD = 0.85


@weave.op()
def planner_structure_check(output: dict) -> dict:
    """Five structural checks on planner output (no ground truth needed).

    1. Valid JSON structure (list of dicts)
    2. Question count in [3, 7]
    3. Each item has both `question` and `source_hint`
    4. No empty strings in either field
    5. No near-duplicate questions (SequenceMatcher > 0.85)
    """
    questions = output.get("sub_questions", [])

    valid_structure = isinstance(questions, list) and all(
        isinstance(q, dict) for q in questions
    )

    count_in_range = 3 <= len(questions) <= 7

    has_both_fields = (
        all("question" in q and "source_hint" in q for q in questions)
        if valid_structure
        else False
    )

    no_empty_strings = (
        all(
            q.get("question", "").strip() and q.get("source_hint", "").strip()
            for q in questions
        )
        if valid_structure
        else False
    )

    texts = [
        q["question"]
        for q in questions
        if isinstance(q, dict) and "question" in q
    ]
    duplicate_pairs: list[tuple[str, str, float]] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ratio = SequenceMatcher(
                None, texts[i].lower(), texts[j].lower()
            ).ratio()
            if ratio > _DUPLICATE_THRESHOLD:
                duplicate_pairs.append((texts[i], texts[j], round(ratio, 3)))
    no_duplicates = len(duplicate_pairs) == 0

    checks = [valid_structure, count_in_range, has_both_fields, no_empty_strings, no_duplicates]
    return {
        "valid_structure": valid_structure,
        "count_in_range": count_in_range,
        "has_both_fields": has_both_fields,
        "no_empty_strings": no_empty_strings,
        "no_duplicates": no_duplicates,
        "structure_score": sum(checks) / len(checks),
        "duplicate_pairs": duplicate_pairs,
    }


# ---------------------------------------------------------------------------
# 3. W/ GT & LLM – Gold-standard concept coverage
# ---------------------------------------------------------------------------


class PlannerGTCoverageScorer(Scorer):
    """LLM judge: count how many gold-standard dimensions the output covers.

    Coverage Score = (# matched GT concepts) / (total GT concepts)
    """

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, target: dict, research_query: str) -> dict:
        generated = output.get("sub_questions", [])
        expected = target.get("expected_sub_questions", [])
        if not expected:
            return {
                "gt_hits": 0,
                "gt_total": 0,
                "gt_coverage": 1.0,
                "missing": [],
                "explanation": "",
            }

        gen_text = "\n".join(
            f"- {q['question']} (source: {q.get('source_hint', 'N/A')})"
            if isinstance(q, dict)
            else f"- {q}"
            for q in generated
        )
        exp_labels = [_extract_question_text(q) for q in expected]
        exp_text = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(exp_labels))

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate whether model-generated research sub-questions "
                        "cover each gold-standard sub-question.\n\n"
                        "For EACH gold-standard question (listed by number), decide "
                        "whether the model output addresses the same concept or "
                        "dimension, even if worded differently. Return 1 if covered, "
                        "0 if not.\n\n"
                        "Return JSON:\n"
                        '{"results": {"1": 0 or 1, "2": 0 or 1, ...}, '
                        '"explanation": "<brief reasoning>"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Model-generated sub-questions:\n{gen_text}\n\n"
                        f"Gold-standard sub-questions:\n{exp_text}"
                    ),
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        results = parsed.get("results", {})

        hits = sum(
            int(bool(results.get(str(i + 1), 0))) for i in range(len(expected))
        )
        missing = [
            exp_labels[i]
            for i in range(len(expected))
            if not results.get(str(i + 1), 0)
        ]

        return {
            "gt_hits": hits,
            "gt_total": len(expected),
            "gt_coverage": hits / len(expected),
            "missing": missing,
            "explanation": parsed.get("explanation", ""),
        }


# ---------------------------------------------------------------------------
# 4. W/o GT & LLM – 6-point binary rubric
# ---------------------------------------------------------------------------

_RUBRIC_CRITERIA = [
    "coverage_completeness",
    "non_overlap",
    "logical_ordering",
    "answerability",
    "source_appropriateness",
    "has_limitation_question",
]


class PlannerRubricScorer(Scorer):
    """LLM judge: 6-point binary rubric (no ground truth needed).

    1. Coverage completeness
    2. Non-overlap
    3. Logical ordering
    4. Answerability
    5. Source appropriateness
    6. Presence of limitation/counterargument question
    """

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, research_query: str) -> dict:
        generated = output.get("sub_questions", [])
        gen_text = "\n".join(
            f"- {q['question']} (source_hint: {q.get('source_hint', 'N/A')})"
            if isinstance(q, dict)
            else f"- {q}"
            for q in generated
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate a set of research sub-questions against a "
                        "6-criterion rubric. For each criterion return exactly "
                        "1 (pass) or 0 (fail).\n\n"
                        "Criteria:\n"
                        "1. coverage_completeness – Do the sub-questions collectively "
                        "cover the full scope of the original research query?\n"
                        "2. non_overlap – Are the sub-questions distinct with minimal "
                        "redundancy?\n"
                        "3. logical_ordering – Are the questions ordered from "
                        "foundational concepts to more specific details?\n"
                        "4. answerability – Can each sub-question reasonably be "
                        "answered with a single web search?\n"
                        "5. source_appropriateness – Does each source_hint "
                        "appropriately match the type of information the question "
                        "seeks?\n"
                        "6. has_limitation_question – Does at least one sub-question "
                        "address limitations, risks, or opposing viewpoints?\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "coverage_completeness": 0 or 1,\n'
                        '  "non_overlap": 0 or 1,\n'
                        '  "logical_ordering": 0 or 1,\n'
                        '  "answerability": 0 or 1,\n'
                        '  "source_appropriateness": 0 or 1,\n'
                        '  "has_limitation_question": 0 or 1,\n'
                        '  "explanation": "<brief reasoning for each criterion>"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Sub-questions:\n{gen_text}"
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
# Registry
# ---------------------------------------------------------------------------


def get_scorers() -> list:
    return [
        planner_semantic_similarity,  # 1. W/ GT  & Code
        planner_structure_check,      # 2. W/o GT & Code
        PlannerGTCoverageScorer(),    # 3. W/ GT  & LLM
        PlannerRubricScorer(),        # 4. W/o GT & LLM
    ]
