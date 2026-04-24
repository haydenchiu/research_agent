from __future__ import annotations

import weave
from tavily import TavilyClient

from config.settings import get_tavily_api_key


@weave.op
def tavily_search(
    query: str,
    *,
    max_results: int = 5,
    search_depth: str = "advanced",
) -> list[dict]:
    """Search the web via Tavily and return structured results."""
    client = TavilyClient(api_key=get_tavily_api_key())
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        include_raw_content=False,
    )
    results = []
    for item in response.get("results", []):
        results.append(
            {
                "query": query,
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "relevance_score": item.get("score", 0.0),
            }
        )
    return results
