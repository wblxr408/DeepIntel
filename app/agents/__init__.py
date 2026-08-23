"""
Agent implementations for the research workflow.

Each agent is a self-contained unit responsible for a specific task:
- PlannerAgent: decomposes queries into execution plans
- SearchAgent: executes web search with query expansion
- BrowserAgent: autonomous web browsing and content extraction
- RAGAgent: hybrid retrieval from the knowledge base
- AnalystAgent: synthesizes evidence into coherent analysis
- ReflectionAgent: validates quality and detects hallucinations
- ReportAgent: generates final Markdown reports
"""

from app.agents.analyst import AnalystAgent
from app.agents.browser import BrowserAgent
from app.agents.planner import PlannerAgent
from app.agents.rag import RAGAgent
from app.agents.reflection import ReflectionAgent
from app.agents.report import ReportAgent
from app.agents.search import SearchAgent

__all__ = [
    "AnalystAgent",
    "BrowserAgent",
    "PlannerAgent",
    "RAGAgent",
    "ReflectionAgent",
    "ReportAgent",
    "SearchAgent",
]
