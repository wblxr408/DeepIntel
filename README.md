# Agentic Deep Research System (ADRS)

> 基于 LangGraph 的自主研究 Agent 平台，强调可执行 DAG、多 Agent 协作、SSE 流式输出、RAG 检索和反思校验。

## 项目定位

这个仓库是一个可展示 Agent Engineering 能力的研究系统。核心链路是：

`用户输入 -> Planner 生成 DAG -> 搜索/浏览/RAG 并行采集 -> Analyst 综合 -> Reflection 校验 -> Report 输出`

## 技术栈

| 层     | 选型                      | 作用                             |
| ------ | ------------------------- | -------------------------------- |
| 编排   | LangGraph                 | 状态机式工作流、条件分支、检查点 |
| 后端   | FastAPI                   | HTTP API、SSE 流式接口           |
| LLM    | OpenAI 兼容接口           | 支持 Qwen / DeepSeek / OpenAI    |
| 检索   | pgvector + BM25 + rerank  | 混合检索与重排序                 |
| 浏览器 | Playwright                | 网页抓取与页面提取               |
| 数据库 | PostgreSQL                | 会话、引用、系统配置             |
| 缓存   | Redis                     | 会话缓存、SSE 状态、运行时配置   |
| 前端   | React + Vite + TypeScript | 研究界面与实时输出               |

## 核心能力

- 自动生成研究 DAG
- 工具驱动的多 Agent 协作
- 长生命周期会话与检查点
- SSE 实时事件流
- 证据驱动的报告生成
- 反思校验与重规划
- 报告长度分档控制（短文 / 中篇 / 长篇）
- 内部知识库源 CRUD、分组管理与文件上传

## 目录结构

```text
deepintel/
├── app/                          # 后端主代码
├── docs/                         # 设计、部署、模型与学习资料
├── frontend/                     # 前端项目
├── metrics/                      # 各能力主题的指标采集
├── scripts/                      # 模型预下载与辅助脚本
├── tests/                        # 单元测试与集成测试
├── docker-compose.yml            # 本地一键启动编排
├── Dockerfile                    # 后端镜像
├── requirements.txt              # Python 依赖
├── start-deepintel.ps1           # Windows 启动脚本
├── SPEC.md                       # 技术规格文档
└── README.md                     # 项目总览
```

### `app/` 文件说明

#### `app/main.py`

- FastAPI 入口
- 初始化日志、中间件、路由和生命周期
- 启动时连接数据库，关闭时释放资源

#### `app/config.py`

- 统一读取环境变量
- 定义 LLM、数据库、Redis、RAG、浏览器、API 配置

#### `app/llm_client.py`

- 构建 OpenAI 兼容客户端
- 统一不同 LLM provider 的接入方式

#### `app/api/`

- `health.py`：健康检查与就绪检查
- `research.py`：研究任务创建、状态查询、结果查询、SSE 流式接口
- `config.py`：前端可配置的 LLM 运行时配置接口
- `documents.py`：内部知识库源管理、分组、上传与切分入库接口

#### `app/agents/`

- `planner.py`：把用户问题拆成 DAG
- `search.py`：搜索 Agent
- `browser.py`：浏览器 Agent
- `rag.py`：RAG Agent
- `analyst.py`：综合分析
- `reflection.py`：事实校验与重规划判断
- `report.py`：报告生成
- `browser_demo.py`：浏览器能力演示

#### `app/graph/`

- `state.py`：LangGraph 状态、DAG、证据、引用、校验模型
- `compiler.py`：把节点和边编译成可执行工作流
- `nodes.py`：节点定义
- `edges.py`：边路由定义

#### `app/rag/`

- `embedder.py`：向量嵌入
- `retriever.py`：混合检索
- `reranker.py`：重排序

#### `app/tools/`

- `search_tools.py`：搜索相关工具封装
- `browser_tools.py`：浏览器操作封装
- `retrieval_tools.py`：检索相关工具封装

#### `app/db/`

- `connection.py`：数据库与 Redis 连接
- `models.py`：数据模型
- `migrate.py`：建表与初始化 schema

#### `app/observability/`

- `sse_manager.py`：管理 SSE 事件队列
- `trace.py`：Agent 事件追踪与结构化日志

### 其他目录说明

#### `frontend/`

- `src/App.tsx`：前端主入口与视图切换
- `src/main.tsx`：React 挂载入口
- `src/components/ResearchDashboard.tsx`：研究工作台
- `src/components/AgentTrace.tsx`：Agent 事件流展示
- `src/components/ToolTrace.tsx`：工具调用流展示
- `src/components/ReportPreview.tsx`：报告预览
- `src/components/LLMConfigPanel.tsx`：LLM 配置面板
- `src/components/DocumentManager.tsx`：知识库源、分组、上传与维护界面
- `src/hooks/useSSE.ts`：SSE 连接封装

#### `metrics/`

- `langgraph_workflow/`：工作流指标
- `research_dag/`：DAG 质量与并行度指标
- `multi_agent/`：多 Agent 协作指标
- `stateful_agent/`：会话与恢复指标
- `browser_agent/`：浏览器执行指标
- `reflection_agent/`：校验与幻觉控制指标

#### `tests/`

- `tests/agents/`：Agent 单测
- `tests/graph/`：图编排单测
- `tests/integration/`：端到端流程测试

#### `scripts/`

- `preload_models.py`：预下载模型
- `download_all_models.py`：一次性下载全部依赖模型

#### `docs/`

- `DEPLOYMENT.md`：部署说明
- `MODEL_DOWNLOAD.md`：模型下载说明
- `LLM_CONFIG_FRONTEND.md`：前端配置说明
- `INTERVIEW_GUIDE.md`：面试讲解资料
- `WBS.md`：工作分解与开发计划

## 开发细节

### 研究流程

1. 用户在前端输入研究主题
2. `app/api/research.py` 创建 session
3. `app/graph/compiler.py` 编译 LangGraph
4. `app/agents/planner.py` 生成 DAG
5. `search / browser / rag` 并行采集证据
6. `analyst.py` 生成分析结论
7. `reflection.py` 判断是否需要重规划
8. `report.py` 生成 Markdown 报告并推送 SSE

### 输出长度控制

- 前端支持 `short / medium / long` 三档输出长度。
- 后端按档位控制报告生成 token 预算、搜索次数和工具结果上限。
- 目标是避免报告被截断，同时让简单问题保持低成本。

### 知识库管理

- 前端提供知识库源的新增、编辑、删除和分组管理。
- 上传文件支持 `json`、`md`、`docx`、`pdf`、`txt`。
- 文档入库时会执行切分和 embedding，再写入向量检索表。

### 状态与持久化

- 研究状态定义在 `app/graph/state.py`
- 会话数据写入 PostgreSQL
- 检查点优先尝试 Redis，失败后回退到内存
- `research_sessions` 保存任务主记录
- `citations` 保存引用明细

### SSE 与前端联动

- 后端通过 `app/observability/sse_manager.py` 维护每个 session 的事件队列
- 前端通过 `frontend/src/hooks/useSSE.ts` 订阅 `/api/v1/research/stream/{session_id}`
- `ResearchDashboard` 组合展示 Agent Trace、Tool Trace 和报告内容

### 配置优先级

运行时配置优先级为：

`前端运行时配置 > 数据库 system_config > 环境变量`

这意味着可以在不改代码的情况下切换 LLM provider 和模型。

## 本地开发

### 环境要求

- Python 3.11+
- Node.js 18+
- PostgreSQL 16 + pgvector
- Redis 7+
- Playwright Chromium

### 后端启动

```bash
.venv\Scripts\activate  # Windows
pip install -r requirements-local.txt
python -m app.db.migrate
uvicorn app.main:app --reload --port 8000
```

### 前端启动

```bash
cd frontend
npm install
npm run dev
```

### Docker 启动

```bash
docker compose up -d
```

生产部署前必须从 `.env.example` 创建私有 `.env`，并设置唯一的
`SECURITY_ENCRYPTION_KEY`、`SECURITY_ENVIRONMENT=production`、
`SECURITY_SECURE_COOKIES=true` 和精确的 `API_CORS_ORIGINS`。生产模式会在
启动时拒绝示例密钥、缺失密钥、非安全 Cookie 或 `*` CORS；反向代理的 HTTPS
不替代应用层登录。持久化密钥加密仅保护数据库/备份泄露，不保护已被攻陷的运行进程。

Windows 也可以用：

```powershell
.\start-deepintel.ps1
```

### 本机启动

```powershell
.\start-deepintel.ps1 -Mode local
```

脚本会优先使用项目内 `.venv`，不存在时自动创建并安装依赖。
如果你想直接使用现成的 conda 环境，可以先设置 `DEEPINTEL_PYTHON=C:\Users\wblxr\anaconda3\envs\used_pytorch\python.exe`。

如果只想启动后端：

```powershell
.\start-deepintel.ps1 -Mode local -SkipFrontend
```

## 环境变量

`.env.example` 已包含完整模板，核心项如下：

```env
LLM_PROVIDER=qwen
LLM_MODEL=qwen-plus
LLM_API_KEY=your-api-key
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

DATABASE_URL=postgresql://deepintel:deepintel_secret@localhost:5433/deepintel
REDIS_URL=redis://localhost:6379/0

RAG_EMBED_MODEL=BAAI/bge-zh-qwen2-int8
RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3
PLAYWRIGHT_HEADLESS=true
API_PORT=8000
FRONTEND_PORT=5173
```

## 开发与调试

- 健康检查：`GET /api/v1/health`
- 就绪检查：`GET /api/v1/ready`
- 研究提交：`POST /api/v1/research`
- 结果查看：`GET /api/v1/research/{session_id}`
- 实时流：`GET /api/v1/research/stream/{session_id}`

建议优先看这几个文件：

- `app/main.py`
- `app/api/research.py`
- `app/graph/compiler.py`
- `app/graph/state.py`
- `frontend/src/components/ResearchDashboard.tsx`

## 学习资料

### 官方文档

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
- [Vite](https://vite.dev/)
- [Playwright Python](https://playwright.dev/python/)
- [PostgreSQL](https://www.postgresql.org/docs/)
- [Redis](https://redis.io/docs/)
- [pgvector](https://github.com/pgvector/pgvector)

### 代码相关资料

- [SPEC.md](SPEC.md)
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- [docs/MODEL_DOWNLOAD.md](docs/MODEL_DOWNLOAD.md)
- [docs/INTERVIEW_GUIDE.md](docs/INTERVIEW_GUIDE.md)

## 测试

```bash
pytest

如果 Windows 上直接执行 `python` / `pytest` 遇到启动器异常，优先使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\scripts\test.ps1
```
cd frontend
npm run build
```

## 备注

- 本项目的 README 已按当前仓库代码整理
- 如果后续新增模块，建议同步补充 `README.md` 和 `docs/`
