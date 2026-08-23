"""
Browser Use Demo - 展示完整的浏览器自动化能力。

运行方式:
    python -m app.agents.browser_demo

本脚本演示 Browser Agent 的 5 大核心能力：

1. OPEN  - 自主打开任意 URL
2. SEARCH - 页面内搜索 / 关键词定位
3. SCROLL - 滚动加载动态内容
4. EXTRACT - 提取结构化数据
5. ANALYZE - AI 理解页面内容

这些能力让 Agent 可以像人一样浏览互联网：
- 打开网页 → 阅读 → 滚动 → 提取 → 理解 → 决策下一步

面试价值：
- 展示了 Agent 不只是调用 API，而是真正自主操作浏览器
- 对比纯 RAG 项目，这是一个显著差异化亮点
- 体现了 LLM + Tool Use + Browser Automation 的结合
"""

from __future__ import annotations

import asyncio
import json

from app.agents.browser import (
    BrowserAgent,
    SmartScroller,
    StructuredExtractor,
    URLDiscovery,
)


async def demo_open_and_browse():
    """
    演示 1: OPEN - 自主打开网页。
    """
    print("\n" + "=" * 70)
    print("演示 1: OPEN - 自主打开任意 URL")
    print("=" * 70)

    agent = BrowserAgent()

    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://github.com/features的人工智能",
        "https://news.ycombinator.com",
    ]

    for url in test_urls:
        print(f"\n>>> 打开: {url}")
        result = await agent.open_and_browse(
            query_or_url=url,
            user_query="AI research",
            extraction_level="deep",
            max_chars=3000,
        )
        print(f"    标题: {result.title}")
        print(f"    类型: {result.page_type.value}")
        print(f"    内容长度: {len(result.extracted_content)} chars")
        print(f"    摘要: {result.extracted_content[:200]}...")

    await agent._cleanup_browser()


async def demo_search_and_navigate():
    """
    演示 2: SEARCH - 搜索查询并导航。
    """
    print("\n" + "=" * 70)
    print("演示 2: SEARCH - 查询转 URL + 智能导航")
    print("=" * 70)

    discovery = URLDiscovery()

    queries = [
        "LangGraph multi-agent workflow",
        "browser use AI agent",
        "DeepSeek R1 reasoning model",
    ]

    for q in queries:
        search_url = discovery.query_to_search_url(q)
        wiki_url = discovery.query_to_wiki_url(q.replace(" ", "_"))

        print(f"\n查询: '{q}'")
        print(f"  搜索引擎: {search_url[:80]}...")
        print(f"  Wikipedia: {wiki_url}")


async def demo_smart_scroll():
    """
    演示 3: SCROLL - 智能滚动加载动态内容。
    """
    print("\n" + "=" * 70)
    print("演示 3: SCROLL - 智能滚动检测")
    print("=" * 70)

    scroller = SmartScroller(max_scrolls=5)

    print(f"""
Smart Scroller 功能:
- 自动检测页面滚动区域
- 滚动直到无新内容加载（最多 {scroller.max_scrolls} 次）
- 自动检测并点击"加载更多"按钮
- 检测页面是否到达底部

滚动策略:
1. 平滑滚动到页面底部
2. 等待内容加载 ({scroller.scroll_delay_ms}ms)
3. 比较高度变化判断是否有新内容
4. 查找并点击"加载更多"按钮
5. 重复直到到达底部或达到最大次数
""")


async def demo_structured_extraction():
    """
    演示 4: EXTRACT - 提取结构化数据。
    """
    print("\n" + "=" * 70)
    print("演示 4: EXTRACT - 结构化数据提取")
    print("=" * 70)

    StructuredExtractor()

    print("""
Structured Extractor 支持的数据类型:

1. 表格提取 (tables)
   - 自动检测 <table> 元素
   - 解析行列数据
   - 最多 5 个表格，每个最多 20 行

2. 列表提取 (lists)
   - 提取 <ul> 和 <ol> 列表项
   - 最多 30 项

3. JSON-LD 提取 (json_ld)
   - 解析 <script type="application/ld+json">
   - 提取结构化数据（Article、Product 等）

4. 元数据提取 (metadata)
   - SEO: description, keywords, author
   - Open Graph: og:title, og:description
   - 文章: article:published_time

5. 链接提取 (links)
   - 提取所有外链
   - 用于页面链式导航

示例数据结构:
""")

    sample_result = {
        "tables": [
            [["Model", "Score", "Params"],
             ["GPT-4", "92.3%", "1.8T"],
             ["Claude 3", "89.1%", "500B"]]
        ],
        "metadata": {
            "description": "Benchmark comparison of LLM models",
            "og:title": "LLM Leaderboard 2026",
            "article:published_time": "2026-01-15",
        },
        "links": [
            {"href": "/gpt-5", "text": "GPT-5 Performance Analysis"},
            {"href": "/claude-4", "text": "Claude 4 Review"},
        ]
    }
    print(json.dumps(sample_result, indent=2, ensure_ascii=False))


async def demo_full_workflow():
    """
    演示 5: 完整的 Browser Use 工作流。

    模拟一个真实研究场景：
    1. 搜索 AI 最新进展
    2. 打开多个相关页面
    3. 提取关键信息
    4. AI 分析页面内容
    """
    print("\n" + "=" * 70)
    print("演示 5: 完整 Browser Use 工作流")
    print("=" * 70)

    print("""
完整 Browser Use 工作流示例:

场景: 研究"2026年 AI Agent 最新进展"

Step 1 OPEN:
    Agent: 收到查询"2026年 AI Agent 最新进展"
    Action: URLDiscovery.query_to_search_url() → Google 搜索 URL

Step 2 SCROLL:
    Agent: 打开搜索结果页
    Action: SmartScroller.auto_scroll() → 滚动加载更多结果

Step 3 EXTRACT:
    Agent: 从每个结果页提取标题、摘要、链接
    Action: StructuredExtractor.extract_all() → 表格/列表/元数据

Step 4 ANALYZE:
    Agent: AI 分析每个页面，决定哪些内容相关
    Action: PageAnalyzer.analyze_and_plan() → 页面类型 + 关键信息 + 下一步

Step 5 NAVIGATE:
    Agent: 点击最相关的链接，继续深入研究
    Action: 重复 Step 1-4，直到收集足够证据

关键设计亮点:
- 三级提取策略管理 token: snippet(1000) → skim(4000) → deep(8000)
- 智能滚动处理动态页面: Twitter/知乎/微博 无限滚动
- 页面内容 AI 分析: 不只是提取，而是理解内容
- 链接自动发现: 提取相关链接用于链式导航
- 并发控制: 浏览器池限制并发，避免被封
""")


async def demo_context_explosion_handling():
    """
    演示: 上下文爆炸问题及其解决方案。
    """
    print("\n" + "=" * 70)
    print("演示: 上下文爆炸问题的处理")
    print("=" * 70)

    print("""
问题背景:
    一个长网页可能有 100,000 tokens。
    如果直接发给 LLM:
    - 成本: $0.01-0.10 per 1000 tokens
    - 上下文: 快速占满 128K 上下文窗口
    - 质量: 无关内容干扰 LLM 理解

本项目的解决方案:
    ┌─────────────────────────────────────────────┐
    │  Query: "AI Agent 的核心架构设计"            │
    └─────────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────┐
    │  Step 1: Snippet (1,000 chars)              │
    │  只提取 meta description + og:title          │
    │  快速判断页面是否相关                         │
    │  成本: ~$0.0001                             │
    └─────────────────────────────────────────────┘
                      │
            ┌─────────┴─────────┐
            │ 相关               │ 不相关
            ▼                   ▼
    ┌────────────────┐
    │  Step 2: Skim  │
    │  (4,000 chars)  │
    │  主要段落提取    │
    │  成本: ~$0.0004 │
    └────────────────┘
            │
            │ 深度相关
            ▼
    ┌────────────────┐
    │  Step 3: Deep   │
    │  (8,000 chars)  │
    │  标题+段落+列表  │
    │  成本: ~$0.0008 │
    └────────────────┘

收益:
    - 减少 90%+ token 消耗
    - 只在真正需要时提取完整内容
    - 对比"直接提取全部内容": 节省 ~$0.009 per page
""")


async def main():
    """运行所有演示。"""
    print("\n" + "=" * 70)
    print("Browser Use Demo - AI Agent 浏览器自动化能力展示")
    print("=" * 70)
    print("""
本演示展示 Browser Agent 的 5 大核心能力：
  1. OPEN   - 自主打开任意 URL
  2. SEARCH - 查询转 URL + 智能导航
  3. SCROLL - 智能滚动加载动态内容
  4. EXTRACT - 提取结构化数据
  5. ANALYZE - AI 理解页面内容

这些能力让 Agent 可以像人一样自主浏览互联网，
对比纯 RAG 项目，这是显著的差异化亮点。
""")

    demos = [
        ("URL Discovery", demo_search_and_navigate),
        ("Smart Scroller", demo_smart_scroll),
        ("Structured Extraction", demo_structured_extraction),
        ("Context Explosion Handling", demo_context_explosion_handling),
        ("Full Workflow", demo_full_workflow),
    ]

    for name, func in demos:
        try:
            await func()
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
            print(f"\n演示 '{name}' 出错: {e}")

    # 真实浏览器演示（需要网络）
    print("\n" + "=" * 70)
    print("真实浏览器演示")
    print("=" * 70)
    print("""
如果需要真实浏览器测试，可以运行以下代码：

    agent = BrowserAgent()
    result = await agent.open_and_browse(
        query_or_url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        user_query="AI research and developments",
        extraction_level="deep",
        max_chars=5000,
    )
    print(result.title, result.extracted_content[:500])

提示: 确保已安装 Playwright:
    pip install playwright
    playwright install chromium
""")


if __name__ == "__main__":
    asyncio.run(main())
