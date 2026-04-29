"""Searcher agent scorers."""

from __future__ import annotations

import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

# ---------------------------------------------------------------------------
# Code / No Ground Truth
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"query", "url", "title", "content"}


@weave.op()
def searcher_structure_check(output: dict) -> dict:
    """Check that all search results have required fields and non-empty content."""
    results = output.get("search_results", [])
    if not results:
        return {"has_results": False, "all_fields_present": False, "all_content_non_empty": False}

    all_fields = all(REQUIRED_FIELDS.issubset(r.keys()) for r in results)
    all_content = all(bool(r.get("content", "").strip()) for r in results)
    return {
        "has_results": True,
        "all_fields_present": all_fields,
        "all_content_non_empty": all_content,
    }


# ---------------------------------------------------------------------------
# Code / With Ground Truth
# ---------------------------------------------------------------------------


@weave.op()
def searcher_url_recall(output: dict, target: dict) -> dict:
    """Check whether expected URLs appear in search results."""
    results = output.get("search_results", [])
    expected_urls = target.get("expected_urls", [])
    if not expected_urls:
        return {"url_recall": 1.0}

    found_urls = {r.get("url", "") for r in results}
    hits = sum(1 for url in expected_urls if url in found_urls)
    return {"url_recall": hits / len(expected_urls)}


# ---------------------------------------------------------------------------
# LLM / No Ground Truth
# ---------------------------------------------------------------------------


class SearcherRelevanceScorer(Scorer):
    """LLM judge: are the search results relevant to the sub-questions? (binary rubric)"""

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, sub_questions: list[str]) -> dict:
        results = output.get("search_results", [])
        results_summary = "\n".join(
            f"- [{r.get('title', '')}] {r.get('content', '')[:200]}" for r in results[:10]
        )
        questions_text = "\n".join(f"- {q}" for q in sub_questions)

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Evaluate search results against the research sub-questions. "
                        "For each criterion, return 1 if satisfied or 0 if not.\n"
                        "- relevance: do the results address the sub-questions?\n"
                        "- coverage: do the results cover most of the sub-questions?\n"
                        "- quality: are the sources substantive (not shallow/empty)?\n"
                        "Return JSON: {\"relevance\": int, \"coverage\": int, "
                        "\"quality\": int, \"explanation\": str}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Sub-questions:\n{questions_text}\n\nSearch results:\n{results_summary}",
                },
            ],
        )
        parsed = parse_json_response(response.choices[0].message.content)
        relevance = int(bool(parsed.get("relevance", 0)))
        coverage = int(bool(parsed.get("coverage", 0)))
        quality = int(bool(parsed.get("quality", 0)))
        return {
            "relevance": relevance,
            "coverage": coverage,
            "quality": quality,
            "score": relevance + coverage + quality,
            "explanation": parsed.get("explanation", ""),
        }


def get_scorers() -> list:
    return [
        searcher_structure_check,
        searcher_url_recall,
        SearcherRelevanceScorer(),
    ]
