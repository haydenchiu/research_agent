from __future__ import annotations

import weave
from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_factory import build_llm
from agents.utils import load_prompt, log_agent, parse_json_response
from graph.state import ResearchState
from tools.search import tavily_search


class SearcherAgent(weave.Model):
    model_name: str = "gpt-4o-mini"
    provider: str = "openai"
    temperature: float = 0.1
    max_tokens: int = 4096

    @weave.op()
    def predict(self, sub_questions: list[str]) -> dict:
        llm = build_llm(self.provider, self.model_name, self.temperature, self.max_tokens)
        system_prompt = load_prompt("searcher")

        all_results: list[dict] = []
        for question in sub_questions:
            results = tavily_search(question, max_results=5)
            all_results.extend(results)

        results_text = _format_results_for_llm(sub_questions, all_results)

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"Sub-questions:\n{_format_questions(sub_questions)}\n\n"
                        f"Raw search results:\n{results_text}"
                    )
                ),
            ]
        )

        parse_json_response(response.content)

        return {"search_results": all_results}

    def node(self, state: ResearchState) -> dict:
        sub_questions = state.get("sub_questions", [])
        result = self.predict(sub_questions=sub_questions)
        result["messages"] = [
            log_agent(
                "searcher",
                f"Collected {len(result['search_results'])} results across {len(sub_questions)} queries",
            )
        ]
        return result


def _format_questions(questions: list[str]) -> str:
    return "\n".join(f"  {i+1}. {q}" for i, q in enumerate(questions))


def _format_results_for_llm(questions: list[str], results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"- [{r['title']}]({r['url']})")
        lines.append(f"  Query: {r['query']}")
        content_preview = r["content"][:500] if r["content"] else "(no content)"
        lines.append(f"  Content: {content_preview}")
        lines.append("")
    return "\n".join(lines)
