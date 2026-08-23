"""
Browser tools for LangChain/LangGraph integration.

Provides Playwright-based web browsing as a LangChain-compatible tool.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def browse_webpage(url: str, max_chars: int = 2000) -> str:
    """
    Browse a webpage and extract its main content. Use this for deep reading
    of articles, official documents, and detailed content that search snippets
    cannot provide.

    This tool:
    - Navigates to the URL
    - Extracts main content (headings, paragraphs)
    - Handles dynamic pages (JavaScript-rendered)
    - Returns structured text

    Args:
        url: The URL to browse (must be a valid http/https URL)
        max_chars: Maximum characters to extract (default 2000)

    Returns:
        Extracted page content as text
    """
    try:
        # Run in new event loop to avoid conflicts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_browse_async(url, max_chars))
        loop.close()
        return result
    except RuntimeError:
        # Already in event loop
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(_browse_async(url, max_chars))
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
            logger.error(f"Browser error for {url}: {e}")
            return f"Error browsing {url}: {e!s}"


async def _browse_async(url: str, max_chars: int) -> str:
    """Async implementation of web browsing."""
    from playwright.async_api import async_playwright

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.set_extra_http_headers({
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            })

            response = await page.goto(url, timeout=30000, wait_until="domcontentloaded")
            if response is None or response.status >= 400:
                await browser.close()
                return f"Page returned status {response.status if response else 'None'}"

            content_parts: list[str] = []

            # Extract title
            title = await page.title()
            if title:
                content_parts.append(f"# {title}\n")

            # Extract headings
            headings = await page.query_selector_all("h1, h2, h3")
            for h in headings[:10]:
                text = await h.inner_text()
                if text and len(text) > 3:
                    content_parts.append(f"## {text.strip()}\n")

            # Extract paragraphs
            paragraphs = await page.query_selector_all("article p, main p, .content p")
            for p in paragraphs[:20]:
                text = await p.inner_text()
                if len(text) > 30:
                    content_parts.append(text.strip() + "\n")
                    if sum(len(c) for c in content_parts) > max_chars:
                        break

            await browser.close()

            result = "\n".join(content_parts)
            return result[:max_chars]

    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
        logger.error(f"Async browse error for {url}: {e}")
        return f"Error: {e!s}"


def get_browser_tools():
    """Return all browser tools for LangGraph."""
    return [browse_webpage]
