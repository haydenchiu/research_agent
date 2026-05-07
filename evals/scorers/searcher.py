"""Searcher agent scorers (3 quadrants of the 2x2 eval matrix).

1. W/ GT  & Code  – Fact Extraction Hit Rate, Source Coverage, URL Validation
2. W/o GT & Code  – JSON Schema Compliance, Citation Linkage, Confidence Distribution,
                    Deduplication Check
3. W/o GT & LLM   – 4-point binary rubric (Source Authority, Contradiction Awareness,
                    Query Efficacy, Conciseness)
"""

from __future__ import annotations

import urllib.error
import urllib.request
from difflib import SequenceMatcher
from urllib.parse import urlparse

import weave
from openai import OpenAI
from weave import Scorer

from agents.utils import parse_json_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUPLICATE_THRESHOLD = 0.85


def _extract_domain(url: str) -> str:
    """Extract the root domain from a URL, stripping 'www.' prefix."""
    domain = urlparse(url).netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.lower()


def _collect_key_facts(findings: list[dict]) -> list[str]:
    """Flatten all key_facts across findings into a single list."""
    facts: list[str] = []
    for f in findings:
        facts.extend(f.get("key_facts", []))
    return facts


def _collect_source_urls(findings: list[dict]) -> list[str]:
    """Flatten all source URLs across findings into a single list."""
    urls: list[str] = []
    for f in findings:
        for src in f.get("sources", []):
            url = src.get("url", "") if isinstance(src, dict) else str(src)
            if url:
                urls.append(url)
    return urls


# ===========================================================================
# 1. W/ GT & Code
# ===========================================================================


@weave.op()
def searcher_fact_extraction_hit_rate(output: dict, target: dict) -> dict:
    """Check if key_facts contain specific mandatory strings/numbers from GT.

    Each expected_fact is checked as a case-insensitive substring match
    against the concatenated key_facts text.
    """
    findings = output.get("findings", [])
    expected_facts = target.get("expected_facts", [])
    if not expected_facts:
        return {"fact_hits": 0, "fact_total": 0, "fact_hit_rate": 1.0, "missing_facts": []}

    all_facts_text = " ".join(_collect_key_facts(findings)).lower()

    missing = []
    hits = 0
    for fact in expected_facts:
        if fact.lower() in all_facts_text:
            hits += 1
        else:
            missing.append(fact)

    return {
        "fact_hits": hits,
        "fact_total": len(expected_facts),
        "fact_hit_rate": hits / len(expected_facts),
        "missing_facts": missing,
    }


@weave.op()
def searcher_source_coverage(output: dict, target: dict) -> dict:
    """Compare found source URLs against a golden list from GT.

    An expected URL counts as covered if it appears exactly in the found set.
    """
    findings = output.get("findings", [])
    golden_urls = target.get("expected_urls", [])
    if not golden_urls:
        return {"source_hits": 0, "source_total": 0, "source_coverage": 1.0, "missing_urls": []}

    found_urls = set(_collect_source_urls(findings))

    missing = []
    hits = 0
    for url in golden_urls:
        if url in found_urls:
            hits += 1
        else:
            missing.append(url)

    return {
        "source_hits": hits,
        "source_total": len(golden_urls),
        "source_coverage": hits / len(golden_urls),
        "missing_urls": missing,
    }


@weave.op()
def searcher_url_validation(output: dict, target: dict) -> dict:
    """Ping source URLs to verify they are live and match GT domains.

    Two sub-scores:
      - live_rate: fraction of source URLs returning HTTP 200
      - domain_match_rate: fraction of source URL domains present in GT expected_domains
    """
    findings = output.get("findings", [])
    source_urls = _collect_source_urls(findings)
    expected_domains = {d.lower() for d in target.get("expected_domains", [])}

    if not source_urls:
        return {
            "live_count": 0,
            "total_urls": 0,
            "live_rate": 0.0,
            "domain_matches": 0,
            "domain_match_rate": 0.0,
        }

    live_count = 0
    domain_matches = 0

    for url in source_urls:
        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "research-agent-eval/1.0")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    live_count += 1
        except Exception:
            pass

        domain = _extract_domain(url)
        if domain and domain in expected_domains:
            domain_matches += 1

    return {
        "live_count": live_count,
        "total_urls": len(source_urls),
        "live_rate": live_count / len(source_urls),
        "domain_matches": domain_matches,
        "domain_match_rate": domain_matches / len(source_urls),
    }


# ===========================================================================
# 2. W/o GT & Code
# ===========================================================================


_FINDINGS_REQUIRED_KEYS = {"sub_question", "key_facts", "sources", "confidence", "contradictions"}
_ALLOWED_CONFIDENCE = {"high", "medium", "low"}


@weave.op()
def searcher_schema_compliance(output: dict) -> dict:
    """Verify every object in findings has the required keys."""
    findings = output.get("findings", [])
    if not findings:
        return {"has_findings": False, "all_keys_present": False, "compliance_details": []}

    details = []
    all_valid = True
    for i, f in enumerate(findings):
        present = set(f.keys()) if isinstance(f, dict) else set()
        missing = _FINDINGS_REQUIRED_KEYS - present
        valid = len(missing) == 0
        if not valid:
            all_valid = False
        details.append({"index": i, "valid": valid, "missing_keys": sorted(missing)})

    return {
        "has_findings": True,
        "all_keys_present": all_valid,
        "compliance_details": details,
    }


@weave.op()
def searcher_citation_linkage(output: dict) -> dict:
    """Verify that for every fact in key_facts, there is at least one source."""
    findings = output.get("findings", [])
    if not findings:
        return {"all_facts_cited": False, "uncited_count": 0, "total_facts": 0}

    total_facts = 0
    uncited = 0
    uncited_examples: list[str] = []

    for f in findings:
        if not isinstance(f, dict):
            continue
        facts = f.get("key_facts", [])
        sources = f.get("sources", [])
        has_sources = len(sources) > 0

        for fact in facts:
            total_facts += 1
            if not has_sources:
                uncited += 1
                if len(uncited_examples) < 5:
                    uncited_examples.append(fact[:100])

    return {
        "all_facts_cited": uncited == 0 and total_facts > 0,
        "uncited_count": uncited,
        "total_facts": total_facts,
        "citation_rate": (total_facts - uncited) / total_facts if total_facts else 0.0,
        "uncited_examples": uncited_examples,
    }


@weave.op()
def searcher_confidence_distribution(output: dict) -> dict:
    """Check that all confidence values use the allowed enum: high, medium, low."""
    findings = output.get("findings", [])
    if not findings:
        return {"all_valid": False, "invalid_values": [], "distribution": {}}

    distribution: dict[str, int] = {}
    invalid_values: list[str] = []

    for f in findings:
        if not isinstance(f, dict):
            continue
        conf = f.get("confidence", "")
        if isinstance(conf, str):
            conf_lower = conf.strip().lower()
        else:
            conf_lower = str(conf)

        distribution[conf_lower] = distribution.get(conf_lower, 0) + 1
        if conf_lower not in _ALLOWED_CONFIDENCE:
            invalid_values.append(conf_lower)

    return {
        "all_valid": len(invalid_values) == 0 and len(distribution) > 0,
        "invalid_values": invalid_values,
        "distribution": distribution,
    }


@weave.op()
def searcher_deduplication_check(output: dict) -> dict:
    """Check for near-duplicate key_facts using SequenceMatcher ratio.

    Pairs with similarity > 0.85 are flagged as duplicates.
    """
    findings = output.get("findings", [])
    all_facts = _collect_key_facts(findings)

    if len(all_facts) < 2:
        return {"no_duplicates": True, "duplicate_pairs": [], "total_facts": len(all_facts)}

    duplicate_pairs: list[dict] = []
    for i in range(len(all_facts)):
        for j in range(i + 1, len(all_facts)):
            ratio = SequenceMatcher(
                None, all_facts[i].lower(), all_facts[j].lower()
            ).ratio()
            if ratio > _DUPLICATE_THRESHOLD:
                duplicate_pairs.append({
                    "fact_a": all_facts[i][:120],
                    "fact_b": all_facts[j][:120],
                    "similarity": round(ratio, 3),
                })

    return {
        "no_duplicates": len(duplicate_pairs) == 0,
        "duplicate_pairs": duplicate_pairs,
        "total_facts": len(all_facts),
    }


# ===========================================================================
# 3. W/o GT & LLM – 4-point binary rubric
# ===========================================================================

_RUBRIC_CRITERIA = [
    "source_authority",
    "contradiction_awareness",
    "query_efficacy",
    "conciseness",
]


class SearcherRubricScorer(Scorer):
    """LLM judge: 4-point binary rubric evaluated without ground truth.

    1. Source Authority – Are sources from authoritative domains?
    2. Contradiction Awareness – Did the agent populate contradictions for
       controversial topics?
    3. Query Efficacy – Were the inferred queries keyword-rich rather than
       overly conversational?
    4. Conciseness – Are key_facts presented as facts/statistics/claims rather
       than long paragraphs?
    """

    model_id: str = "gpt-4o-mini"

    @weave.op()
    def score(self, output: dict, sub_questions: list[str], research_query: str) -> dict:
        findings = output.get("findings", [])
        findings_text = _format_findings_for_judge(findings)
        questions_text = "\n".join(f"- {q}" for q in sub_questions)

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate search agent output against a 4-criterion "
                        "rubric. For each criterion return exactly 1 (pass) or "
                        "0 (fail).\n\n"
                        "Criteria:\n"
                        "1. source_authority - Are the cited source titles and "
                        "URLs from recognized authoritative domains (e.g. .gov, "
                        ".edu, major news outlets, established research "
                        "institutions) rather than low-quality blogs or "
                        "unrecognizable sites? Score 1 if the MAJORITY of "
                        "sources are authoritative.\n"
                        "2. contradiction_awareness - If the research topic is "
                        "controversial or has multiple viewpoints, did the agent "
                        "populate the `contradictions` field with meaningful "
                        "entries? Score 1 if contradictions are noted where "
                        "appropriate, or if the topic is uncontroversial and "
                        "empty contradictions are acceptable.\n"
                        "3. query_efficacy - Looking at the sub-questions and "
                        "the sub_question field in each finding, are the "
                        "queries keyword-rich and optimized for web search "
                        "rather than overly conversational or vague? Score 1 if "
                        "the majority are well-formulated search queries.\n"
                        "4. conciseness - Are the key_facts presented as "
                        "concise facts, statistics, and claims rather than "
                        "long, rambling paragraphs? Score 1 if each fact is a "
                        "focused, digestible statement.\n\n"
                        "Return JSON:\n"
                        "{\n"
                        '  "source_authority": 0 or 1,\n'
                        '  "contradiction_awareness": 0 or 1,\n'
                        '  "query_efficacy": 0 or 1,\n'
                        '  "conciseness": 0 or 1,\n'
                        '  "explanation": "<brief reasoning for each criterion>"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research query: {research_query}\n\n"
                        f"Sub-questions given to agent:\n{questions_text}\n\n"
                        f"Agent findings output:\n{findings_text}"
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


def _format_findings_for_judge(findings: list[dict]) -> str:
    """Render findings into a readable text block for the LLM judge."""
    if not findings:
        return "(no findings)"
    lines: list[str] = []
    for i, f in enumerate(findings):
        if not isinstance(f, dict):
            continue
        lines.append(f"Finding {i + 1}:")
        lines.append(f"  sub_question: {f.get('sub_question', 'N/A')}")
        facts = f.get("key_facts", [])
        lines.append(f"  key_facts ({len(facts)}):")
        for fact in facts[:10]:
            lines.append(f"    - {fact[:300]}")
        sources = f.get("sources", [])
        lines.append(f"  sources ({len(sources)}):")
        for src in sources[:5]:
            if isinstance(src, dict):
                lines.append(f"    - [{src.get('title', '')}] {src.get('url', '')}")
            else:
                lines.append(f"    - {src}")
        lines.append(f"  confidence: {f.get('confidence', 'N/A')}")
        contras = f.get("contradictions", [])
        lines.append(f"  contradictions ({len(contras)}):")
        for c in contras[:5]:
            lines.append(f"    - {c[:200]}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_scorers() -> list:
    return [
        # 1. W/ GT & Code
        searcher_fact_extraction_hit_rate,
        searcher_source_coverage,
        searcher_url_validation,
        # 2. W/o GT & Code
        searcher_schema_compliance,
        searcher_citation_linkage,
        searcher_confidence_distribution,
        searcher_deduplication_check,
        # 3. W/o GT & LLM
        SearcherRubricScorer(),
    ]
