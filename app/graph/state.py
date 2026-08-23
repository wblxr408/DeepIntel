"""
ResearchState definition for LangGraph StateGraph.

===============================================================
五大核心主题在这里的映射
===============================================================

主题 1 - Autonomous Research Workflow:
    - ResearchState.task_status 追踪整个工作流状态
    - 所有节点共享同一个 state，实现工作流端到端

主题 2 - Research DAG Generation:
    - PlanNode/PlanEdge/DAGDefinition 模型
    - research_plan 从静态列表升级为可执行的 DAG
    - 节点支持 depends_on 依赖关系
    - 支持 parallel 并行标记

主题 3 - Tool-driven Multi-Agent Collaboration:
    - ToolCallRecord 追踪每次工具调用
    - 每个子 Agent 有独立的 tool_calls 列表
    - 工具调用有明确的 input_schema 和 output_schema

主题 4 - Long-running Stateful Agent:
    - checkpoint_id 追踪当前检查点
    - session_metadata 存储会话元数据
    - revision_count 追踪重规划次数
    - PostgresSaver 保存完整状态快照

主题 5 - Self-Reflection & Verification:
    - ReflectionResult 模型
    - VerificationDimension 枚举
    - 每个声明有 confidence_score
    - evidence_citations 记录证据引用

===============================================================
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from app.core.time import utc_now_naive


def merge_lists(left: list | None, right: list | None) -> list:
    """Merge list state without treating dict entries as chat messages."""
    if not left:
        return list(right or [])
    if not right:
        return list(left)
    return [*left, *right]


# ==============================================================
# Enums - 对应各主题中的角色定义
# ==============================================================

class AgentType(str, Enum):
    """工具驱动多 Agent 的角色枚举（主题 3）"""
    PLANNER = "planner"       # DAG 生成器
    SEARCH = "search"          # 搜索工具使用者
    BROWSER = "browser"       # 浏览器工具使用者
    RAG = "rag"              # 检索工具使用者
    ANALYST = "analyst"       # 分析工具使用者（内部推理）
    REFLECTION = "reflection"  # 自校验器（主题 5）
    REPORT = "report"        # 报告生成器


class StepStatus(str, Enum):
    """DAG 节点状态（主题 2）"""
    PENDING = "pending"       # 未开始
    RUNNING = "running"       # 执行中
    DONE = "done"           # 已完成
    FAILED = "failed"        # 失败
    SKIPPED = "skipped"      # 跳过（被条件分支排除）


class TaskStatus(str, Enum):
    """自主研究工作流整体状态（主题 1）"""
    PENDING = "pending"       # 等待开始
    RUNNING = "running"       # 执行中
    COMPLETED = "completed"   # 成功完成
    FAILED = "failed"         # 失败
    PAUSED = "paused"        # 暂停（等待人工介入）


class RuntimeStatus(str, Enum):
    """细粒度运行时状态，用于治理闭环和兼容映射。"""
    PENDING = "pending"
    APPROVED = "approved"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    RETRYABLE_FAILED = "retryable_failed"
    TERMINAL_FAILED = "terminal_failed"
    COMPLETED = "completed"
    CANCELED = "canceled"


class PageType(str, Enum):
    """浏览器提取的页面类型（主题 3 工具设计）"""
    NEWS_ARTICLE = "news_article"
    TECHNICAL = "technical"
    SEARCH_RESULT = "search_result"
    SOCIAL = "social"
    GENERAL = "general"


class VerificationDimension(str, Enum):
    """自校验维度（主题 5）"""
    FACTUALITY = "factuality"           # 事实性
    NUMERICAL_ACCURACY = "numerical"     # 数值准确性
    TEMPORAL_VALIDITY = "temporal"       # 时效性
    SELF_CONSISTENCY = "consistency"     # 自洽性
    COMPLETENESS = "completeness"        # 完整性
    CITATION_COVERAGE = "citation"        # 引用覆盖率


# ==============================================================
# 主题 2 - Research DAG Generation: DAG 结构模型
# ==============================================================

class PlanNode(BaseModel):
    """
    DAG 中的单个节点（研究步骤）。

    对应主题 2：不再是静态的研究步骤列表，而是有依赖关系的有向图节点。
    """
    node_id: str = Field(default_factory=lambda: f"n{uuid.uuid4().hex[:6]}")
    # 节点类型：对应工具驱动中的 Agent 类型
    node_type: Literal["search", "browser", "rag", "mcp", "analyst", "reflection", "report"] | None = None
    # 该节点要执行的具体查询
    query: str = ""
    # 依赖的前置节点（只有在这些节点完成后才能执行）
    depends_on: list[str] = Field(default_factory=list)
    # 是否可以并行执行（与 depends_on 互斥）
    parallel: bool = True
    # 状态
    status: StepStatus = StepStatus.PENDING
    # 重试次数
    retry_count: int = 0
    max_retries: int = 2
    # 超时时间（秒）
    timeout_seconds: int = 300
    # 执行结果
    result: dict | None = None
    # 置信度
    confidence: float = 0.0
    # 最近一次错误
    last_error: str | None = None
    last_error_category: str | None = None
    # 是否为终止性失败
    terminal_failure: bool = False
    # 是否等待审批
    waiting_approval: bool = False

    # ===== 向后兼容字段 =====
    # 兼容旧的 PlanStep 的字段名
    step_id: str | None = Field(default=None)
    description: str | None = Field(default=None)
    assigned_agent: str | None = Field(default=None)
    target_query: str | None = Field(default=None)
    evidence_ids: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @property
    def search_query(self) -> str:
        """兼容旧接口：返回 target_query 或 query"""
        return self.target_query or self.query

    @property
    def agent_type(self) -> str:
        """兼容旧接口：返回 assigned_agent 或 node_type"""
        return self.assigned_agent or self.node_type

    def model_post_init(self, *args, **kwargs) -> None:
        """自动填充兼容字段"""
        # 如果 query 为空但 target_query 有值，则用 target_query
        if not self.query and self.target_query:
            object.__setattr__(self, "query", self.target_query)
        # 如果 node_type 为空但 assigned_agent 有值，则用 assigned_agent
        if not self.node_type and self.assigned_agent:
            object.__setattr__(self, "node_type", self.assigned_agent)
        # step_id alias
        if not self.step_id:
            object.__setattr__(self, "step_id", self.node_id)
        if not self.node_type or not self.query:
            raise ValueError("PlanNode requires either node_type/query or assigned_agent/target_query")


class PlanEdge(BaseModel):
    """
    DAG 中的边（节点间关系）。

    对应主题 2：定义节点之间的执行顺序和条件。
    """
    from_node: str
    to_node: str
    # 边的类型：顺序依赖 / 条件分支
    edge_type: Literal["sequential", "conditional"] = "sequential"
    # 条件表达式（用于条件分支边）
    condition: str | None = None


class DAGDefinition(BaseModel):
    """
    完整的研究 DAG 定义。

    对应主题 2：Planner Agent 生成的"可执行的研究计划图"。
    这是研究计划从"静态配置"到"可执行代码"的升级。
    """
    dag_id: str = Field(default_factory=lambda: f"dag-{uuid.uuid4().hex[:8]}")
    dag_name: str
    created_at: str = Field(default_factory=lambda: utc_now_naive().isoformat())
    nodes: list[PlanNode] = Field(default_factory=list)
    edges: list[PlanEdge] = Field(default_factory=list)

    def get_executable_order(self) -> list[list[str]]:
        """
        计算可执行的节点顺序。

        使用拓扑排序 + Kahn 算法：
        1. 找出所有入度为 0 的节点（第一批可并行执行）
        2. 移除这批节点，更新其他节点的入度
        3. 重复直到所有节点处理完

        返回：[[可并行节点列表], ...]
        """
        # 构建邻接表和入度表
        in_degree = {n.node_id: len(n.depends_on) for n in self.nodes}
        adj = {n.node_id: [] for n in self.nodes}
        for edge in self.edges:
            adj[edge.from_node].append(edge.to_node)

        result = []
        processed = set()
        current_batch = [n.node_id for n in self.nodes if in_degree[n.node_id] == 0]

        while current_batch:
            result.append(current_batch)
            processed.update(current_batch)

            # 更新入度
            next_batch = []
            for node_id in current_batch:
                for neighbor in adj[node_id]:
                    if neighbor not in processed:
                        # 检查所有前置节点是否都已处理
                        node = next((n for n in self.nodes if n.node_id == neighbor), None)
                        if (
                            node
                            and all(dep in processed for dep in node.depends_on)
                            and neighbor not in next_batch
                        ):
                            next_batch.append(neighbor)

            current_batch = next_batch

        return result

    def to_json(self) -> dict:
        """导出为 JSON，用于可视化和调试。"""
        return {
            "dag_id": self.dag_id,
            "dag_name": self.dag_name,
            "nodes": [n.model_dump() for n in self.nodes],
            "edges": [e.model_dump() for e in self.edges],
            "execution_order": self.get_executable_order(),
        }


# ==============================================================
# 主题 3 - Tool-driven Multi-Agent: 工具调用追踪
# ==============================================================

class ToolCallRecord(BaseModel):
    """
    单次工具调用的完整记录。

    对应主题 3：追踪每个 Agent 的每个工具调用，
    实现完整的工具调用审计。
    """
    call_id: str = Field(default_factory=lambda: f"call-{uuid.uuid4().hex[:8]}")
    # 调用者
    agent_type: AgentType
    # 工具名称
    tool_name: str
    # 输入参数（脱敏处理）
    args: dict[str, Any] = Field(default_factory=dict)
    # 调用时间
    started_at: str = Field(default_factory=lambda: utc_now_naive().isoformat())
    completed_at: str | None = None
    duration_ms: int | None = None
    # 状态
    status: Literal["pending", "running", "success", "error", "timeout"] = "pending"
    # 结果摘要
    result_summary: str | None = None
    # 安全治理元数据（来源、脱敏、审批等）
    safety_json: dict[str, Any] = Field(default_factory=dict)
    # 错误信息
    error: str | None = None
    # 成本
    cost_usd: float = 0.0
    # token 消耗
    tokens_used: int = 0


class ToolInvocationHistory(BaseModel):
    """
    单个 Agent 的工具调用历史。

    对应主题 3：每个子 Agent 有独立的工具调用记录，
    便于分析和优化工具使用效率。
    """
    agent_type: AgentType
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def success_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.status == "success")

    @property
    def error_count(self) -> int:
        return sum(1 for tc in self.tool_calls if tc.status == "error")

    @property
    def efficiency(self) -> float:
        """工具调用效率：成功调用 / 总调用"""
        if self.total_calls == 0:
            return 0.0
        return self.success_count / self.total_calls

    @property
    def total_duration_ms(self) -> int:
        return sum(tc.duration_ms or 0 for tc in self.tool_calls)


class NodeOutcome(BaseModel):
    """单个 DAG 工具节点的执行结果。"""
    node_id: str
    tool_call_id: str | None = None
    tool_name: str
    status: Literal["success", "retryable_error", "terminal_error", "awaiting_approval", "skipped"]
    error_category: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    result_count: int = 0
    approval_request_id: str | None = None


# ==============================================================
# 主题 4 - Long-running Stateful Agent: 状态与检查点
# ==============================================================

class SessionMetadata(BaseModel):
    """
    会话元数据（主题 4）。

    对应主题 4：长生命周期 Agent 的会话上下文，
    存储在检查点中，支持会话恢复。
    """
    session_id: str
    user_query: str
    created_at: str = Field(default_factory=lambda: utc_now_naive().isoformat())
    updated_at: str = Field(default_factory=lambda: utc_now_naive().isoformat())
    completed_at: str | None = None
    # 当前检查点 ID
    checkpoint_id: str | None = None
    # 状态
    status: TaskStatus = TaskStatus.PENDING
    # 重规划计数
    revision_count: int = 0
    max_revisions: int = 3
    # 成本追踪
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    # 错误记录
    error_count: int = 0
    last_error: str | None = None


# ==============================================================
# 主题 5 - Self-Reflection & Verification: 校验模型
# ==============================================================

class ClaimEvidence(BaseModel):
    """
    声明与证据的关联（主题 5）。

    对应主题 5：每个声明都关联到其支撑证据，
    实现可追溯的推理链。
    """
    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    is_verified: bool = False
    verification_dimensions: list[VerificationDimension] = Field(default_factory=list)


class Citation(BaseModel):
    """引用记录"""
    citation_id: str = Field(default_factory=lambda: f"cit-{uuid.uuid4().hex[:8]}")
    text: str = ""
    source_url: str | None = None
    source_title: str | None = None
    source_type: Literal["web", "document", "knowledge_base"] = "web"
    # 附加元数据（兼容各 Agent 的字段）
    extracted_evidence: str | None = None
    relevance_score: float = 0.0


class SearchResult(BaseModel):
    """搜索引擎结果"""
    url: str
    title: str
    snippet: str
    relevance_score: float = Field(ge=0.0, default=0.0)
    published_date: str | None = None
    domain: str | None = None


class BrowserResult(BaseModel):
    """浏览器提取结果"""
    url: str
    title: str
    extracted_content: str
    page_type: PageType = PageType.GENERAL
    citations: list[str] = Field(default_factory=list)
    tokens_extracted: int = 0
    # 兼容 browser.py 传入的字段
    extraction_level: str = "skim"  # snippet | skim | deep
    citation: str | None = None  # 单条引用文本
    error_message: str | None = None


class RAGResult(BaseModel):
    """RAG 检索结果"""
    chunk_id: str
    content: str
    score: float = Field(ge=0.0, default=0.0)
    metadata: dict = Field(default_factory=dict)
    citation: str | None = None
    # 兼容 rag.py 传入的评分字段
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


# ==============================================================
# 主题 5 - Self-Reflection & Verification: 校验模型
# ==============================================================

class HallucinatedClaim(BaseModel):
    """
    幻觉声明（主题 5）。

    对应主题 5：Reflection Agent 检测出的问题声明。
    """
    claim: str
    severity: Literal["high", "medium", "low"] = "medium"
    reason: str
    suggested_fix: str = ""  # 兼容 reflection.py 的 suggested_action


class ClaimConflict(BaseModel):
    """
    声明间的逻辑冲突（主题 5）。

    对应主题 5：检测出相互矛盾的声明。
    """
    claim_a: str
    claim_b: str
    conflict_description: str


class VerificationResult(BaseModel):
    """
    完整校验结果（主题 5）。

    对应主题 5：Reflection Agent 的输出结构，
    用于决定是否需要重规划。
    """
    total_claims: int = 0
    verified_claims: int = 0
    hallucinated_claims: list[HallucinatedClaim] = Field(default_factory=list)
    conflicts: list[ClaimConflict] = Field(default_factory=list)
    factuality_score: float = Field(ge=0.0, le=1.0, default=0.5)
    consistency_score: float = Field(ge=0.0, le=1.0, default=0.5)
    completeness_score: float = Field(ge=0.0, le=1.0, default=0.5)
    citation_coverage: float = Field(ge=0.0, le=1.0, default=0.5)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    needs_revision: bool = False
    revision_focus: str | None = None

    @property
    def hallucination_rate(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return len(self.hallucinated_claims) / self.total_claims

    @property
    def passes_threshold(self) -> bool:
        return (
            self.overall_confidence >= 0.85
            and self.citation_coverage >= 0.95
            and self.hallucination_rate < 0.05
            and len(self.conflicts) == 0
        )


# VerificationResult 别名（向后兼容）
ReflectionResult = VerificationResult


class ErrorRecord(BaseModel):
    """错误记录"""
    error_id: str = Field(default_factory=lambda: f"err-{uuid.uuid4().hex[:8]}")
    agent: str
    error_type: str
    message: str
    timestamp: str = Field(default_factory=lambda: utc_now_naive().isoformat())
    recoverable: bool = True


# ==============================================================
# 主题 3 - 工具驱动多 Agent: 证据模型
# ==============================================================

class Evidence(BaseModel):
    """
    研究证据（主题 3/5）。

    对应主题 3：工具调用收集的原始证据
    对应主题 5：用于校验和引用追踪
    """
    evidence_id: str = Field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:8]}")
    content: str
    source_url: str | None = None
    source_title: str | None = None
    source_type: Literal["web", "document", "knowledge_base"] = "web"
    # 来源 Agent
    collected_by: AgentType
    collected_at: str = Field(default_factory=lambda: utc_now_naive().isoformat())
    # 可信度
    reliability: float = Field(ge=0.0, le=1.0, default=0.7)
    # Token 消耗估计
    tokens_estimate: int = 0

    @property
    def agent_type(self) -> AgentType:
        """Backward-compatible alias for older agent_type readers."""
        return self.collected_by


# ==============================================================
# TypedDict State
# ==============================================================

class ResearchState(TypedDict):
    """
    LangGraph 工作流的核心状态（主题 1）。

    包含五大主题的所有状态：
    - task_status（主题 1）
    - dag（主题 2）
    - tool_histories（主题 3）
    - session（主题 4）
    - verification（主题 5）
    """

    # ===== 主题 1 & 4: 任务与会话 =====
    task_id: str
    user_query: str
    created_at: str
    status: str                        # TaskStatus
    session: dict                    # SessionMetadata

    # ===== 主题 2: DAG =====
    dag: dict | None                # DAGDefinition (序列化)
    current_executing_nodes: list[str]  # 当前执行的节点
    completed_nodes: list[str]        # 已完成节点
    node_outcomes: Annotated[list[dict], merge_lists]

    # ===== 主题 3: 工具调用历史 =====
    tool_histories: Annotated[list[dict], merge_lists]  # ToolInvocationHistory[]
    collected_evidence: Annotated[list[dict], merge_lists]  # Evidence[]

    # ===== 兼容旧版字段 =====
    search_results: list[dict]
    browser_results: list[dict]
    rag_results: list[dict]
    aggregated_evidence: list[dict]

    # ===== 主题 5: 校验 =====
    verification: dict | None       # VerificationResult
    revision_needed: bool
    revision_count: int

    # ===== 输出 =====
    analysis: str
    final_report: str
    citations: list[dict]
    guardrail_decision: dict | None
    evidence_status: dict | None
    review_status: dict | None
    failure_memory: dict | None
    user_confirmed: bool

    # ===== 可观测性 =====
    agent_trace: Annotated[list[dict], merge_lists]
    guardrail_trace: Annotated[list[dict], merge_lists]
    errors: list[dict]
    """
    LangGraph 工作流的核心状态（主题 1）。

    包含五大主题的所有状态：
    - task_status（主题 1）
    - dag（主题 2）
    - tool_histories（主题 3）
    - session（主题 4）
    - verification（主题 5）
    """

    # ===== 主题 1 & 4: 任务与会话 =====

    # ===== 主题 2: DAG =====

    # ===== 主题 3: 工具调用历史 =====

    # ===== 兼容旧版字段 =====
    # backward compat: legacy search/browser/rag results (used by analyst_node)

    # ===== 主题 5: 校验 =====

    # ===== 输出 =====
    allow_web_after_rag_hit: bool
    rag_group: str | None
    retrieval_policy: dict | None
    runtime_status: str
    budget_state: dict | None
    pending_approvals: Annotated[list[dict], merge_lists]
    output_length: str
    skill_context: dict | None

    # ===== 可观测性 =====


# ==============================================================
# State Factory
# ==============================================================

def create_initial_state(
    user_query: str,
    task_id: str | None = None,
) -> ResearchState:
    """
    创建初始研究状态（主题 1 & 4）。

    工厂函数，对应主题 1：自主工作流的起点
    对应主题 4：长生命周期会话的初始化
    """
    now = utc_now_naive().isoformat()
    session_id = task_id or str(uuid.uuid4())

    session_meta = SessionMetadata(
        session_id=session_id,
        user_query=user_query,
    )

    return ResearchState(
        task_id=session_id,
        user_query=user_query,
        created_at=now,
        status=TaskStatus.PENDING.value,
        session=session_meta.model_dump(),

        dag=None,
        current_executing_nodes=[],
        completed_nodes=[],
        node_outcomes=[],

        tool_histories=[],
        collected_evidence=[],
        search_results=[],
        browser_results=[],
        rag_results=[],
        aggregated_evidence=[],

        verification=None,
        revision_needed=False,
        revision_count=0,

        analysis="",
        final_report="",
        citations=[],
        guardrail_decision=None,
        evidence_status=None,
        review_status=None,
        failure_memory=None,
        user_confirmed=False,
        allow_web_after_rag_hit=False,
        rag_group=None,
        retrieval_policy=None,
        runtime_status=RuntimeStatus.PENDING.value,
        budget_state=None,
        pending_approvals=[],
        output_length="medium",
        skill_context=None,

        agent_trace=[],
        guardrail_trace=[],
        errors=[],
    )


def serialize_dag(dag: DAGDefinition) -> dict:
    """将 DAGDefinition 序列化为 dict 用于 TypedDict 存储。"""
    return dag.model_dump()


def deserialize_dag(data: dict) -> DAGDefinition:
    """从 dict 反序列化为 DAGDefinition。"""
    return DAGDefinition.model_validate(data)


# ==============================================================
# Backward Compatibility Helpers
# ==============================================================

def deserialize_steps(data: list[dict]) -> list[PlanNode]:
    """将 dict 列表反序列化为 PlanNode 列表（向后兼容）。"""
    return [PlanNode.model_validate(d) for d in data]


def deserialize_evidence(data: dict) -> Evidence:
    """将 dict 反序列化为 Evidence。"""
    return Evidence.model_validate(data)


# ==============================================================
# Legacy Aliases for backward compatibility
# ==============================================================

class AgentEvent(BaseModel):
    """
    Agent 执行事件（向后兼容）。

    用于 agent_trace，记录每个 Agent 的执行事件。
    """
    agent: str
    event_type: str  # agent_start / agent_complete / tool_start / tool_complete / error
    content: str
    timestamp: str = Field(default_factory=lambda: utc_now_naive().isoformat())


# PlanStep 是 PlanNode 的别名（向后兼容旧代码）
PlanStep = PlanNode
__all__ = [
    "AgentEvent",
    "AgentType",
    "ClaimConflict",
    "ClaimEvidence",
    "DAGDefinition",
    "Evidence",
    "HallucinatedClaim",
    "NodeOutcome",
    "PageType",
    "PlanEdge",
    "PlanNode",
    "PlanStep",  # alias for backward compat
    "ReflectionResult",  # alias
    "ResearchState",
    "RuntimeStatus",
    "SessionMetadata",
    "StepStatus",
    "TaskStatus",
    "ToolCallRecord",
    "ToolInvocationHistory",
    "VerificationDimension",
    "VerificationResult",
    "create_initial_state",
    "deserialize_dag",
    "deserialize_evidence",
    "deserialize_steps",
    "serialize_dag",
]

