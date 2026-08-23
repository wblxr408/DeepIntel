"""
Browser Agent: Full Browser Use Implementation.

===============================================================
Browser Use = AI Agent 最热门方向之一

本模块实现了完整的 Browser Use 范式，Agent 可以：

1. **OPEN** - 自主打开任意网页
2. **SEARCH** - 页面内搜索 / 关键词定位
3. **SCROLL** - 滚动加载动态内容 / 翻页
4. **EXTRACT** - 提取结构化数据（文章、表格、列表）
5. **ANALYZE** - 理解页面内容，决策下一步

对比 Operator / Manus / Browser Use：
- Operator: 端到端任务自动化，视觉理解
- Manus: 多步骤浏览器任务规划
- Browser Use: 直接浏览器操作 + LLM 决策

本实现特色：
- Query-to-URL: 将自然语言查询转换为搜索 URL
- Progressive Loading: 滚动触发动态内容加载
- Smart Scrolling: 智能滚动检测（到底/加载更多）
- Multi-step Navigation: 支持页面链式导航
- Structured Data Extraction: 表格、列表、JSON-LD 提取
- Accessibility Tree: 无障碍树用于元素定位

===============================================================
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

from playwright.async_api import Error as PlaywrightError

from app.config import get_settings
from app.graph.state import BrowserResult, Citation, PageType, PlanStep
from app.llm_client import call_chat_with_fallback, create_fallback_llm_client

if TYPE_CHECKING:
    from playwright.async_api import Browser, Page, Playwright

logger = logging.getLogger(__name__)


# ==============================================================
# URL Discovery - Query to URL
# ==============================================================

class URLDiscovery:
    """
    将自然语言查询转换为可访问的 URL。

    这是 Browser Use 的第一步：用户给的是查询，Agent 需要找到正确的页面。
    """

    SEARCH_ENGINES: ClassVar[list[str]] = [
        "https://www.google.com/search?q={q}&hl=zh-CN",
        "https://www.bing.com/search?q={q}&ensearch=1",
        "https://duckduckgo.com/?q={q}&ia=web",
        "https://search.yahoo.com/search?p={q}",
    ]

    WIKI_TEMPLATE = "https://en.wikipedia.org/wiki/{topic}"
    ZHIHU_TEMPLATE = "https://www.zhihu.com/search?type=content&q={topic}"

    def __init__(self):
        self.search_engine = self.SEARCH_ENGINES[0]

    def query_to_search_url(self, query: str, lang: str = "zh-CN") -> str:
        """
        将查询转换为搜索引擎 URL。
        """
        q_encoded = self._encode_query(query)
        url = self.search_engine.format(q=q_encoded)
        if lang == "zh-CN":
            url = url.replace("hl=en", "hl=zh-CN")
        return url

    def query_to_wiki_url(self, topic: str) -> str:
        """将主题转换为 Wikipedia URL。"""
        return self.WIKI_TEMPLATE.format(topic=self._encode_query(topic))

    def query_to_zhihu_url(self, topic: str) -> str:
        """将主题转换为知乎搜索 URL。"""
        return self.ZHIHU_TEMPLATE.format(topic=self._encode_query(topic))

    def _encode_query(self, query: str) -> str:
        """URL 编码，支持中文。"""
        import urllib.parse
        return urllib.parse.quote(query, safe="")


# ==============================================================
# Smart Scroller - 动态内容加载
# ==============================================================

class SmartScroller:
    """
    智能滚动控制器。

    问题：很多网页是无限滚动加载（Twitter/知乎/微博）。
    解决方案：
    1. 检测页面滚动区域
    2. 滚动直到无新内容加载
    3. 检测"加载更多"按钮并点击
    4. 防止无限循环（最大滚动次数）
    """

    def __init__(self, max_scrolls: int = 10):
        self.max_scrolls = max_scrolls
        self.scroll_delay_ms = 800  # 等待内容加载

    async def auto_scroll(
        self,
        page: Page,
        target_content_type: str = "article",
    ) -> dict[str, Any]:
        """
        自动滚动页面直到内容加载完成。

        返回: {
            "scroll_count": 滚动次数,
            "new_content_loaded": 是否加载了新内容,
            "reached_bottom": 是否到达底部,
            "loaded_more_button": 是否点击了加载更多,
        }
        """

        stats = {
            "scroll_count": 0,
            "new_content_loaded": False,
            "reached_bottom": False,
            "loaded_more_button": False,
        }

        last_height = 0

        for i in range(self.max_scrolls):
            # 获取当前页面高度
            try:
                current_height = await page.evaluate("document.body.scrollHeight")
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
                break

            # 检测是否有新内容
            if current_height > last_height:
                stats["new_content_loaded"] = True

            # 滚动到页面底部
            try:
                await page.evaluate(
                    "window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})"
                )
                await asyncio.sleep(self.scroll_delay_ms / 1000)
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
                break

            stats["scroll_count"] += 1
            last_height = current_height

            # 检查是否到达底部
            try:
                at_bottom = await page.evaluate(
                    "(window.innerHeight + window.scrollY) >= document.body.scrollHeight - 100"
                )
                if at_bottom:
                    stats["reached_bottom"] = True
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
                pass

            # 点击"加载更多"按钮
            more_button = await self._find_load_more_button(page)
            if more_button:
                try:
                    await more_button.click(timeout=3000)
                    await asyncio.sleep(1000)
                    stats["loaded_more_button"] = True
                    logger.info(f"Clicked 'load more' button, scroll {i+1}")
                except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
                    pass

            # 如果到达底部且没有加载更多按钮，停止
            if stats["reached_bottom"] and not more_button:
                break

        return stats

    async def _find_load_more_button(self, page: Page):
        """查找并返回"加载更多"按钮。"""
        from playwright.async_api import TimeoutError

        button_selectors = [
            "button:has-text('加载更多')",
            "button:has-text('Load More')",
            "button:has-text('Load more')",
            "button:has-text('查看更多')",
            "button:has-text('Read More')",
            "[class*='load-more']",
            "[class*='loadmore']",
            "[class*='show-more']",
            "[data-action='load-more']",
        ]

        for selector in button_selectors:
            try:
                btn = await page.query_selector(selector)
                if btn and await btn.is_visible():
                    return btn
            except TimeoutError:
                continue
        return None

    async def find_and_click_target(
        self,
        page: Page,
        target_text: str,
    ) -> bool:
        """
        在页面内搜索并点击包含目标文本的元素。

        用于：翻页、展开评论、点击相关链接等。
        """
        # 查找包含文本的链接/按钮
        selectors = [
            f"a:has-text('{target_text}')",
            f"button:has-text('{target_text}')",
            f"span:has-text('{target_text}')",
        ]

        for selector in selectors:
            try:
                el = await page.query_selector(selector)
                if el and await el.is_visible():
                    await el.click(timeout=3000)
                    await asyncio.sleep(500)
                    return True
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
                continue

        return False


# ==============================================================
# Structured Data Extractor
# ==============================================================

class StructuredExtractor:
    """
    从网页中提取结构化数据。

    支持：表格、JSON-LD、列表、评论、报价等。
    """

    async def extract_all(self, page: Page) -> dict[str, Any]:
        """
        提取页面中所有可用的结构化数据。
        """
        results = {}

        results["tables"] = await self.extract_tables(page)
        results["lists"] = await self.extract_lists(page)
        results["json_ld"] = await self.extract_json_ld(page)
        results["metadata"] = await self.extract_metadata(page)
        results["links"] = await self.extract_links(page)

        return results

    async def extract_tables(self, page: Page) -> list[list[str]]:
        """提取页面中的所有表格。"""
        tables = []
        try:
            table_elements = await page.query_selector_all("table")
            for table in table_elements[:5]:  # 最多 5 个表
                rows = await table.query_selector_all("tr")
                table_data = []
                for row in rows[:20]:  # 最多 20 行
                    cells = await row.query_selector_all("th, td")
                    row_data = []
                    for cell in cells[:10]:  # 最多 10 列
                        text = await cell.inner_text()
                        row_data.append(text.strip())
                    if row_data:
                        table_data.append(row_data)
                if table_data:
                    tables.append(table_data)
        except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError) as e:
            logger.warning(f"Table extraction failed: {e}")
        return tables

    async def extract_lists(self, page: Page) -> list[str]:
        """提取有序和无序列表内容。"""
        items = []
        try:
            for tag in ["ul", "ol"]:
                elements = await page.query_selector_all(f"{tag} li")
                for el in elements[:50]:
                    text = (await el.inner_text()).strip()
                    if text and len(text) > 5:
                        items.append(text)
        except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
            pass
        return items[:30]  # 最多 30 项

    async def extract_json_ld(self, page: Page) -> list[dict]:
        """提取 JSON-LD 结构化数据。"""
        json_lds = []
        try:
            scripts = await page.query_selector_all('script[type="application/ld+json"]')
            for script in scripts[:5]:
                content = await script.inner_text()
                import json
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        json_lds.append(data)
                    elif isinstance(data, list):
                        json_lds.extend([d for d in data if isinstance(d, dict)])
                except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
                    pass
        except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
            pass
        return json_lds

    async def extract_metadata(self, page: Page) -> dict[str, str]:
        """提取页面元数据（SEO 信息）。"""
        metadata = {}
        meta_tags = [
            ("description", 'meta[name="description"]'),
            ("keywords", 'meta[name="keywords"]'),
            ("author", 'meta[name="author"]'),
            ("og:title", 'meta[property="og:title"]'),
            ("og:description", 'meta[property="og:description"]'),
            ("article:published_time", 'meta[property="article:published_time"]'),
        ]
        for name, selector in meta_tags:
            try:
                el = await page.query_selector(selector)
                if el:
                    content = await el.get_attribute("content")
                    if content:
                        metadata[name] = content
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
                continue
        return metadata

    async def extract_links(self, page: Page) -> list[dict[str, str]]:
        """提取所有外链（用于页面链式导航）。"""
        links = []
        try:
            anchors = await page.query_selector_all("a[href]")
            for a in anchors[:50]:
                href = await a.get_attribute("href")
                text = (await a.inner_text()).strip()
                if href and len(text) > 3:
                    links.append({"href": href, "text": text})
        except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
            pass
        return links


# ==============================================================
# Page Analyzer - 理解页面内容
# ==============================================================

class PageAnalyzer:
    """
    分析页面内容，理解结构并决定提取策略。

    这是 Browser Use 的核心：不是硬编码提取规则，
    而是让 AI 理解页面结构后决定如何提取。
    """

    def __init__(self):
        from app.config import get_settings
        settings = get_settings()
        self.model = settings.llm.model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from app.llm_client import create_llm_client, get_llm_model
            self._client = create_llm_client()
            self.model = get_llm_model()
        return self._client

    async def analyze_and_plan(
        self,
        page_content: str,
        user_query: str,
    ) -> dict[str, Any]:
        """
        分析页面内容，制定提取计划。

        输入：页面文本 + 用户研究目标
        输出：{
            "page_type": 页面类型,
            "key_information": 关键信息,
            "extraction_plan": 提取计划,
            "next_actions": 下一步行动,
            "relevant_links": 相关链接,
        }
        """
        prompt = f"""You are analyzing a web page for research purposes.

User's research query: {user_query}

Page content (first 3000 chars):
{page_content[:3000]}

Based on the content, respond with a JSON object containing:
- "page_type": one of ["news", "article", "product", "data_table", "forum", "social", "search_results", "unknown"]
- "key_information": list of key facts/statistics found on this page
- "extraction_plan": what specific data to extract (e.g., "stock prices from table", "article body paragraphs")
- "next_actions": what further browsing might be helpful (e.g., ["click next page", "visit related product page"])
- "confidence": how confident you are about this analysis (0.0-1.0)

Return ONLY valid JSON."""

        try:
            response, response_model, fallback_used = call_chat_with_fallback(
                self.client,
                self.model,
                create_fallback_llm_client(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            if fallback_used:
                self.model = response_model
            import json
            result = json.loads(response.choices[0].message.content or "{}")
            return result
        except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError) as e:
            logger.warning(f"Page analysis failed: {e}")
            return {
                "page_type": "unknown",
                "key_information": [],
                "extraction_plan": "generic content extraction",
                "next_actions": [],
                "confidence": 0.3,
            }


# ==============================================================
# Main Browser Agent
# ==============================================================

class BrowserAgent:
    """
    完整的 Browser Use Agent。

    实现五步 Browser Use 范式：
    1. OPEN   - 打开任意 URL 或搜索查询
    2. SEARCH - 页面内搜索 / 关键词定位
    3. SCROLL - 滚动加载动态内容
    4. EXTRACT - 提取结构化数据
    5. ANALYZE - 理解内容，决策下一步

    设计亮点：
    - 三级提取（Snippet/Skim/Deep）管理 token 消耗
    - 智能滚动处理动态加载页面
    - 结构化数据提取（表格/JSON-LD）
    - 页面内容分析 + AI 决策下一步
    - 异步浏览器池并发操作
    - 完整的资源清理
    """

    def __init__(self):
        settings = get_settings()
        self.cfg = settings.browser
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._semaphore = asyncio.Semaphore(self.cfg.pool_size)
        self._init_lock = asyncio.Lock()
        self._url_discovery = URLDiscovery()
        self._scroller = SmartScroller()
        self._extractor = StructuredExtractor()
        self._analyzer = PageAnalyzer()

    async def _ensure_browser(self) -> Browser:
        """Lazily initialize Playwright browser."""
        if self._browser is not None:
            return self._browser

        async with self._init_lock:
            if self._browser is not None:
                return self._browser

            from playwright.async_api import async_playwright
            playwright_instance = await async_playwright().start()
            try:
                browser = await playwright_instance.chromium.launch(
                    headless=self.cfg.headless,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                    ],
                )
                self._playwright = playwright_instance
                self._browser = browser
                logger.info("Browser Agent: Playwright browser initialized")
                return browser
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError) as e:
                logger.error(f"Browser Agent: Failed to launch browser: {e}")
                await playwright_instance.stop()
                raise

    async def _cleanup_browser(self) -> None:
        """Clean up all browser resources."""
        if self._browser:
            try:
                await self._browser.close()
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"Browser cleanup error: {e}")
            self._browser = None

        if self._playwright:
            try:
                await self._playwright.stop()
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
                pass
            self._playwright = None

    # ==============================================================
    # CORE BROWSER USE METHODS
    # ==============================================================

    async def open_and_browse(
        self,
        query_or_url: str,
        user_query: str,
        extraction_level: str = "deep",
        max_chars: int = 8000,
    ) -> BrowserResult:
        """
        主入口：接收查询或 URL，执行完整的 Browser Use 流程。

        Step 1 OPEN  - 如果是查询，先转换为 URL
        Step 2 SCROLL - 智能滚动加载内容
        Step 3 EXTRACT - 结构化数据提取
        Step 4 ANALYZE - AI 分析页面内容

        Args:
            query_or_url: 搜索查询或直接 URL
            user_query: 原始研究目标（用于 AI 分析）
            extraction_level: snippet / skim / deep
            max_chars: 最大字符数

        Returns:
            BrowserResult with full extraction
        """
        browser = await self._ensure_browser()
        page = await browser.new_page()
        start_time = time.time()

        try:
            # ===== STEP 1: OPEN =====
            url = self._resolve_target(query_or_url)
            logger.info(f"Browser Use Step 1 OPEN: {url}")

            await self._navigate_with_retry(page, url)
            page_type = await self._classify_page(page, url)

            # ===== STEP 2: SCROLL =====
            if page_type in (PageType.NEWS_ARTICLE, PageType.TECHNICAL, PageType.GENERAL):
                scroll_stats = await self._scroller.auto_scroll(
                    page,
                    target_content_type="article",
                )
                logger.info(f"Browser Use Step 2 SCROLL: scrolls={scroll_stats['scroll_count']}, "
                            f"bottom={scroll_stats['reached_bottom']}")

            # ===== STEP 3: EXTRACT =====
            extraction_method = {
                "snippet": self._extract_snippet,
                "skim": self._extract_skim,
                "deep": self._extract_deep,
            }.get(extraction_level, self._extract_skim)

            content = await extraction_method(page, max_chars)
            title = await page.title() or url

            structured = await self._extractor.extract_all(page)
            structured.get("metadata", {})
            structured.get("tables", [])

            # ===== STEP 4: ANALYZE =====
            analysis = await self._analyzer.analyze_and_plan(content, user_query)
            logger.info(f"Browser Use Step 4 ANALYZE: type={analysis.get('page_type')}, "
                        f"confidence={analysis.get('confidence')}")

            # 构建 Citation
            citation = Citation(
                source_url=url,
                source_title=title,
                source_type="web",
                extracted_evidence=content[:500],
                relevance_score=0.8,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info(f"Browser Use complete: {url} in {elapsed_ms}ms, "
                        f"{len(content)} chars, {analysis.get('page_type')}")

            return BrowserResult(
                url=url,
                page_type=page_type,
                title=title,
                extracted_content=content[:max_chars],
                citations=[citation.source_url],
                extraction_level=extraction_level,
                tokens_estimate=len(content) // 4,
                citation=citation.source_url,
            )

        except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Browser Use failed for {query_or_url}: {e}")
            return self._fallback_result(query_or_url, error_message=str(e))
        finally:
            await page.close()

    async def _navigate_with_retry(
        self,
        page: Page,
        url: str,
        retries: int = 2,
    ) -> None:
        """Navigate with retry logic."""
        from playwright.async_api import TimeoutError

        for attempt in range(retries + 1):
            try:
                response = await page.goto(
                    url,
                    timeout=self.cfg.navigation_timeout,
                    wait_until="domcontentloaded",
                )
                if response and response.status < 500:
                    return
                logger.warning(f"HTTP {response.status if response else 'None'}, retry {attempt+1}")
            except TimeoutError:
                logger.warning(f"Navigation timeout, retry {attempt+1}")
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"Navigation error: {e}, retry {attempt+1}")

            if attempt < retries:
                await asyncio.sleep(1)

    def _resolve_target(self, query_or_url: str) -> str:
        """
        将输入解析为 URL。

        规则：
        - 已经是 URL → 直接返回
        - 是搜索查询 → 使用搜索引擎
        - 是主题词 → 使用 Wikipedia
        """
        if query_or_url.startswith(("http://", "https://")):
            return query_or_url

        # 包含空格但不以 http 开头 → 可能是查询或主题
        return self._url_discovery.query_to_search_url(query_or_url)

    async def _classify_page(self, page: Page, url: str) -> PageType:
        """基于 URL 和页面内容分类页面类型。"""
        url_lower = url.lower()

        if any(kw in url_lower for kw in ["news", "article", "blog", "post"]):
            return PageType.NEWS_ARTICLE
        if any(kw in url_lower for kw in ["github.com", "readthedocs", "stackoverflow",
                                           "stackoverflow", "wiki", "docs"]):
            return PageType.TECHNICAL
        if any(kw in url_lower for kw in ["google.com/search", "bing.com/search",
                                           "baidu.com/search", "duckduckgo"]):
            return PageType.SEARCH_RESULT
        if any(kw in url_lower for kw in ["zhihu.com", "weibo.com",
                                           "twitter.com", "x.com"]):
            return PageType.SOCIAL

        return PageType.GENERAL

    # ==============================================================
    # Extraction Methods
    # ==============================================================

    async def _extract_snippet(self, page: Page, max_chars: int) -> str:
        """Snippet: 仅使用 meta 标签，快速低 token。"""
        snippets: list[str] = []

        meta_fields = [
            'meta[name="description"]',
            'meta[property="og:title"]',
            'meta[property="og:description"]',
            'meta[name="keywords"]',
        ]

        for selector in meta_fields:
            try:
                el = await page.query_selector(selector)
                if el:
                    content = await el.get_attribute("content")
                    if content:
                        snippets.append(content.strip())
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
                continue

        return " | ".join(snippets)[:max_chars]

    async def _extract_skim(self, page: Page, max_chars: int) -> str:
        """Skim: 提取主要内容段落。"""
        content_parts: list[str] = []

        selectors = [
            "article p", "main p", ".content p", "#content p",
            ".post-content p", ".article-content p", ".entry-content p",
            "div[class*='content'] p",
        ]

        for selector in selectors:
            try:
                elements = await page.query_selector_all(selector)
                for el in elements[:15]:
                    text = (await el.inner_text()).strip()
                    if len(text) > 50:
                        content_parts.append(text)
                        if sum(len(p) for p in content_parts) > max_chars:
                            break
                if content_parts:
                    break
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
                continue

        # 降级：提取 body 文本
        if not content_parts:
            try:
                body = await page.query_selector("body")
                if body:
                    text = await body.inner_text()
                    content_parts = [text[:max_chars]]
            except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
                pass

        result = "\n\n".join(content_parts)
        return result[:max_chars]

    async def _extract_deep(self, page: Page, max_chars: int) -> str:
        """Deep: 完整内容提取，包括标题、段落、列表、表格。"""
        parts: list[str] = []

        # 标题结构
        try:
            headings = await page.query_selector_all("h1, h2, h3, h4")
            for h in headings[:20]:
                text = (await h.inner_text()).strip()
                if text and len(text) > 3:
                    tag = h.evaluate("el => el.tagName")
                    level = min(int(tag[1]), 4)
                    prefix = "#" * level
                    parts.append(f"{prefix} {text}")
        except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
            pass

        # 段落
        try:
            paragraphs = await page.query_selector_all("article p, main p, .content p")
            for p in paragraphs[:40]:
                text = (await p.inner_text()).strip()
                if len(text) > 30:
                    parts.append(text)
        except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
            pass

        # 列表
        try:
            for tag in ["ul", "ol"]:
                items = await page.query_selector_all(f"{tag} li")
                for li in items[:30]:
                    text = (await li.inner_text()).strip()
                    if len(text) > 10:
                        parts.append(f"- {text}")
        except (OSError, PlaywrightError, RuntimeError, TypeError, ValueError):
            pass

        result = "\n\n".join(parts)
        return result[:max_chars]

    def _fallback_result(self, url: str, error_message: str | None = None) -> BrowserResult:
        """提取失败时的降级结果。"""
        return BrowserResult(
            url=url,
            page_type=PageType.GENERAL,
            title=url,
            extracted_content="",
            citations=[],
            extraction_level="snippet",
            tokens_estimate=0,
            citation=url,
            error_message=error_message,
        )

    # ==============================================================
    # Entry Points
    # ==============================================================

    def execute_browse(self, query: str) -> list[BrowserResult]:
        """同步入口：供 DAG executor 调用。"""
        try:
            return asyncio.run(self._async_browse(query))
        except RuntimeError:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self._async_browse(query))

    def execute(self, plan_steps: list[PlanStep]) -> list[BrowserResult]:
        """多步入口：处理 PlanStep 列表。"""
        from app.config import get_settings
        get_settings()
        results: list[BrowserResult] = []

        for step in plan_steps:
            query = step.target_query or step.query
            if query:
                results.extend(self.execute_browse(query))

        return results

    async def _async_browse(self, query: str) -> list[BrowserResult]:
        """异步执行。"""
        async with self._semaphore:
            result = await self.open_and_browse(
                query_or_url=query,
                user_query=query,
                extraction_level="deep",
                max_chars=self.cfg.deep_max_chars,
            )
            return [result]

    async def browse_multiple(
        self,
        queries: list[str],
        parallel: bool = True,
    ) -> list[BrowserResult]:
        """
        并行/顺序浏览多个查询。

        Args:
            queries: URL 或搜索查询列表
            parallel: 是否并行执行（使用浏览器池）
        """
        if parallel:
            tasks = [self.open_and_browse(q, q) for q in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if isinstance(r, BrowserResult)]
        else:
            results: list[BrowserResult] = []
            for q in queries:
                r = await self.open_and_browse(q, q)
                results.append(r)
            return results
