"""
Search Agent: performs web search with query expansion and deduplication.

Implements multi-query expansion and result ranking.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, unquote, urlparse

from app.config import get_settings
from app.graph.state import PlanStep, SearchResult
from app.llm_client import call_chat_with_fallback, create_fallback_llm_client

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


class SearchAgent:
    """
    Handles search query execution with query rewriting and deduplication.

    Expands each plan step into multiple search queries, executes them,
    then deduplicates and ranks the results.
    """

    EXPAND_PROMPT = """Given the original research query, generate 3-5 diverse search queries covering:
1. Core facts and hard data (numbers, statistics)
2. Expert analysis and authoritative opinions
3. Latest news and developments (past 6 months)

Return JSON with a "queries" array of strings."""

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm.model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            from app.llm_client import create_llm_client, get_llm_model
            self._client = create_llm_client()
            self.model = get_llm_model()
        return self._client

    def execute(self, plan_steps: list[PlanStep], user_query: str) -> list[SearchResult]:
        """
        Execute search for given plan steps.

        Args:
            plan_steps: Steps assigned to 'search' agent
            user_query: Original user query for context

        Returns:
            List of deduplicated SearchResult objects
        """
        logger.info(f"Search Agent: executing {len(plan_steps)} plan steps")

        all_results: list[SearchResult] = []

        for step in plan_steps:
            try:
                # Step 1: Expand the query
                expanded_queries = self._expand_query(step.target_query, user_query)

                # Step 2: Execute each expanded query
                for query in expanded_queries:
                    results = self._execute_search(query)
                    all_results.extend(results)
                    time.sleep(0.5)  # Rate limiting

            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
                logger.error(f"Search step error for '{step.target_query}': {e}")
                continue

        # Step 3: Deduplicate and rank
        deduplicated = self._deduplicate(all_results)
        ranked = self._rank_results(deduplicated, user_query)

        logger.info(f"Search Agent: {len(ranked)} unique results after ranking")
        return ranked[:15]  # Top 15 results

    def _expand_query(self, query: str, user_query: str) -> list[str]:
        """Expand a query into multiple search queries."""
        messages = [
            {"role": "system", "content": self.EXPAND_PROMPT},
            {"role": "user", "content": f"Original query: {query}\nContext: {user_query}"},
        ]

        try:
            response, response_model, fallback_used = call_chat_with_fallback(
                self.client,
                self.model,
                create_fallback_llm_client(),
                messages=messages,
                temperature=0.3,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            if fallback_used:
                self.model = response_model
            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                queries = data.get("queries", [])
                if queries:
                    return queries[:5]
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
            logger.warning(f"Query expansion failed: {e}")

        return [query]  # Fallback to original

    def _execute_search(self, query: str) -> list[SearchResult]:
        """
        Execute a single search query.

        Uses DuckDuckGo HTML search for real web results, with the Instant
        Answer API only as a supplemental fallback.
        """
        html_results = self._execute_duckduckgo_html_search(query)
        if html_results:
            return html_results
        return self._execute_duckduckgo_instant_answer(query)

    def _execute_duckduckgo_html_search(self, query: str) -> list[SearchResult]:
        """Fetch and parse DuckDuckGo's HTML results page."""
        try:
            import requests

            url = "https://html.duckduckgo.com/html/"
            params = {
                "q": query,
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.warning("DuckDuckGo HTML search returned status %s", response.status_code)
                return []

            return self._parse_duckduckgo_html(response.text, query)

        except (OSError, RuntimeError, TypeError, ValueError, KeyError, AttributeError) as e:
            logger.warning(f"DuckDuckGo HTML search failed for '{query}': {e}")
            return []

    def _execute_duckduckgo_instant_answer(self, query: str) -> list[SearchResult]:
        """Fallback to DuckDuckGo Instant Answer API."""
        try:
            import requests

            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return []

            data = response.json()
            results: list[SearchResult] = []

            # Extract related topics
            for topic in data.get("RelatedTopics", [])[:8]:
                if "Text" in topic and "URL" in topic:
                    results.append(SearchResult(
                        url=topic["URL"],
                        title=self._clean_html(data.get("Heading", query)),
                        snippet=self._clean_html(topic["Text"])[:300],
                        relevance_score=0.5,
                    ))

            # Extract abstract
            if data.get("AbstractText"):
                results.insert(0, SearchResult(
                    url=data.get("AbstractURL", ""),
                    title=data.get("Heading", query),
                    snippet=data["AbstractText"][:300],
                    relevance_score=0.9,
                ))

            return results

        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
            logger.warning(f"DuckDuckGo instant answer failed for '{query}': {e}")
            return []

    def _parse_duckduckgo_html(self, html_text: str, query: str) -> list[SearchResult]:
        """Parse title, URL and snippet from DuckDuckGo HTML without extra deps."""
        import html as html_lib

        results: list[SearchResult] = []
        blocks = re.findall(
            r'<div[^>]+class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*</div>',
            html_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not blocks:
            blocks = html_text.split('class="result__body"')

        for block in blocks:
            link_match = re.search(
                r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not link_match:
                continue

            raw_url = html_lib.unescape(link_match.group(1))
            title = self._clean_html(link_match.group(2))
            snippet_match = re.search(
                r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>|'
                r'<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>',
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            snippet_html = ""
            if snippet_match:
                snippet_html = next((g for g in snippet_match.groups() if g), "")
            snippet = self._clean_html(snippet_html)
            url = self._normalize_duckduckgo_url(raw_url)

            if not url or not title:
                continue

            parsed = urlparse(url)
            results.append(SearchResult(
                url=url,
                title=title,
                snippet=snippet[:300] or title,
                relevance_score=0.7,
                domain=parsed.netloc,
            ))

            if len(results) >= 10:
                break

        if not results:
            logger.info("DuckDuckGo HTML search parsed 0 results for query: %s", query)
        return results

    def _normalize_duckduckgo_url(self, raw_url: str) -> str:
        """Resolve DuckDuckGo redirect URLs to target URLs."""
        if raw_url.startswith("//"):
            raw_url = f"https:{raw_url}"
        parsed = urlparse(raw_url)
        qs = parse_qs(parsed.query)
        if qs.get("uddg"):
            return unquote(qs["uddg"][0])
        return raw_url

    def _clean_html(self, text: str) -> str:
        """Remove HTML entities and excess whitespace."""
        import html
        import re
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate URLs, keeping highest-scoring entry."""
        seen: dict[str, SearchResult] = {}
        for r in results:
            if r.url and r.url not in seen or r.url in seen and r.relevance_score > seen[r.url].relevance_score:
                seen[r.url] = r
        return list(seen.values())

    def _rank_results(self, results: list[SearchResult], user_query: str) -> list[SearchResult]:
        """Re-rank results by relevance to the user query."""
        keywords = set(user_query.lower().split())
        for r in results:
            score = r.relevance_score
            title_words = set(r.title.lower().split())
            snippet_words = set(r.snippet.lower().split())
            overlap = len(keywords & title_words) + 0.5 * len(keywords & snippet_words)
            r.relevance_score = score + 0.1 * overlap
        return sorted(results, key=lambda r: r.relevance_score, reverse=True)

    def execute_search(self, query: str) -> list[SearchResult]:
        """
        Execute a single search query (主题 3 工具接口).

        直接执行单个搜索查询，供 DAG executor 调用。
        """
        return self._execute_search(query)

    def execute_for_node(self, node_query: str, context: str = "") -> list[SearchResult]:
        """兼容接口：根据节点 query 执行搜索。"""
        return self.execute_search(node_query)
