# RAG_PROJECT — 基于 LangGraph + Milvus 的自纠错知识库系统

一个面向学术文献（当前领域：**铁死亡 / 三阴性乳腺癌 / TCGA-GEO 公共数据集分析 / LASSO 预后模型**）的 RAG 知识库。从 PDF 的 OCR 解析、结构化切分、混合检索一路做到 LangGraph 动态路由 + 自评估重写闭环，最终作为 FastAPI HTTP 服务以 Docker 形式交付给上游 ReAct Agent 调用。

---

## 项目亮点

### 1. 向量库与部署架构
- **Milvus 2.6.x** 作为向量数据库，通过 Docker 启动单节点实例即可运行；生产可横向扩展为多节点集群。
- 整个 API 层（FastAPI + LangGraph）本身也通过 `Dockerfile` + `docker-compose.yml` 打包发布，容器内通过 `host.docker.internal` 访问宿主 Milvus，部署一条命令即可完成。

### 2. 文档解析与结构化
- **PDF → 本地部署 `glm-ocr:bf16` 视觉模型（Ollama 承载）→ Markdown** 的 OCR 管线，输出带标题层级的结构化 Markdown，并落盘缓存为同名 `.md` 文件，二次入库直接走缓存跳过 OCR。
- Markdown 解析使用 `UnstructuredMarkdownLoader` 的 `elements` + `hi_res` 策略抽取标题 / 段落 / 表格等元素。
- 通过自写的 `merge_title_content` 逻辑按 `parent_id` → `element_id` 重建「标题 → 内容」父子关系，避免原始元素被打散后脱离上下文。

### 3. 智能分块策略
- 先做**章节级合并**（title + content 拼成一块），再对超长块（>8000 字符）调用 `SemanticChunker` 按语义断点二次切分。
- 组合策略兼顾语义完整度与单块粒度，缓解跨页断裂和表格割裂问题。

### 4. 混合检索与索引调优
- 同一个 Milvus collection 里同时建 **Dense（HNSW, dim=4096, M=16, efConstruction=64）** 和 **Sparse BM25（jieba + cnalphanumonly filter）** 两路索引。
- 稠密向量使用 `qwen3-embedding:8b`（Ollama 部署，内部通过 OpenAI 兼容协议调用）。
- 稀疏检索走 Milvus 内建的 `BM25BuiltInFunction`，索引算法 **DAAT_MAXSCORE**，`bm25_k1=1.6`, `bm25_b=0.75`。
- 检索阶段使用 **RRF（Reciprocal Rank Fusion）** 融合两路结果，`k=5`，只返 `category == 'content'` 字段。

### 5. 动态路由工作流（LangGraph）
项目提供两套 LangGraph 工作流：

| Graph | 定位 | 流程 |
|---|---|---|
| **graph / Graph 1 — Corrective RAG** | 消息态 Agent 风格 | `agent → retrieve → grade_documents →` 分支 `generate` / `rewrite` |
| **graph2 / Graph 2 — Adaptive RAG** | 状态字典，生产主力 | `route_question →` 分支到 `retrieve` 或 `web_search` → `grade_documents → decide_to_generate → generate →` 幻觉检查 + 答案相关性打分 → 通过则结束，不通过则 `transform_query` 重写或再次 `generate` |

- 路由节点由 LLM 结构化输出判定走**向量库**还是**网络搜索**。
- `decide_to_generate` 在没有相关文档时先走查询重写；超过 `MAX_TRANSFORM_RETRIES=1` 后切到 web_search，避免在空库上无限循环烧 token。

### 6. 闭环评估与纠错机制
- **检索后评估**：`grade_documents` 对每个候选文档做二值相关性打分，不相关的直接过滤。
- **生成后双重评估**：
  - `hallucination_grader`：生成内容是否基于检索到的文档（防幻觉）。
  - `answer_grader`：回答是否真正解决用户问题。
- 幻觉检查不通过 → 最多重试 `MAX_GENERATE_RETRIES=2` 次；答案相关性不过关 → 走 `transform_query` 改写重试。

### 7. 接入上游 Agent
- 项目以 **FastAPI 的 `POST /query`** 为统一入口，返回 `{answer, route, documents, error}`。
- 上游 `ReActAgents` 通过一个 LangChain `@tool(response_format="content_and_artifact")` 工具调用本服务，终端 Agent 只看到简短状态（`KB_OK: ...` / `KB_EMPTY` / `KB_ERROR: timeout` 等），原始文档走 `artifact` 字段不进 LLM 上下文，避免 token 雪球。

---

## 技术栈

| 层 | 组件 |
|---|---|
| Agent / Workflow | LangGraph 1.1.x、LangChain 1.2.x、LangChain Core 1.2.30 |
| RAG | Milvus 2.6.x（Dense HNSW + Sparse BM25）、`langchain-milvus` 0.3.3 |
| LLM | glm-5（tokenhub.tencentmaas.com，外部 API） |
| Embedding | Qwen3-Embedding-8B（ModelScope API / 本地 Ollama 两路） |
| OCR | `glm-ocr:bf16`（Ollama 私有部署） |
| 文档解析 | `pypdfium2` + `unstructured` + `SemanticChunker` |
| 服务 | FastAPI + Uvicorn、Gradio（调试 UI） |
| 部署 | Docker、docker-compose |

---

## 目录结构

```
RAG_PROJECT/
├── api/                    # FastAPI 服务
│   ├── server.py           # /health, /query 端点
│   └── schemas.py          # Pydantic 请求/响应模型
├── graph/                  # Graph 1: Corrective RAG (CLI)
├── graph2/                 # Graph 2: Adaptive RAG (生产路径)
│   ├── graph_2.py          # 主入口，编排所有节点
│   ├── retriever_node.py   # Milvus 混合检索
│   ├── grade_documents_node.py
│   ├── generate_node.py
│   ├── transform_query_node.py
│   ├── web_search_node.py  # Tavily fallback
│   ├── query_route_chain.py      # 路由分流 prompt
│   ├── grade_answer_chain.py     # 答案相关性打分
│   └── grade_hallucinations_chain.py # 幻觉检测
├── documents/              # 解析与入库
│   ├── pdf_parser.py       # PDF → OCR → Markdown
│   ├── markdown_parser.py  # Markdown 元素解析 + 语义切分
│   ├── milvus_db.py        # collection schema / 写入 / 增量去重
│   ├── write_milvus.py     # Markdown 批量入库（多进程）
│   └── write_pdf_milvus.py # PDF 批量入库（多进程）
├── tools/
│   └── retriever_tools.py  # LangChain RetrieverTool（RRF 混合检索）
├── llm_models/
│   ├── api_llm.py          # 外部 API：glm-5 + Qwen3-Embedding-8B + Tavily
│   ├── all_llm.py          # 内网 llama.cpp 通道（legacy/CLI）
│   └── embeddings_model.py # Ollama qwen3-embedding:8b
├── utils/                  # 日志 / env / 打印辅助
├── datas/                  # 语料目录
│   ├── pdf/                # 原始 PDF
│   └── md/                 # Markdown 语料（可选）
├── debug_ui.py             # Gradio 调试界面
├── Dockerfile              # API 服务镜像
├── docker-compose.yml      # API 服务编排
├── requirements-api.txt    # API 子集依赖（容器构建）
└── requirements.txt        # 完整依赖（含解析链所需的 unstructured/pypdfium2 等）
```

---

## 环境准备

### 1. 配置 `.env`

项目根目录创建 `.env`（已在 `.gitignore` / `.dockerignore` 中）：

```bash
# Milvus
MILVUS_URI=http://localhost:19530
COLLECTION_NAME=t_collection01

# 外部 LLM（glm-5）
API_LLM_BASE_URL=https://tokenhub.tencentmaas.com/v1
API_LLM_API_KEY=sk-xxx
API_LLM_MODEL=glm-5

# 外部 Embedding（Qwen3-Embedding-8B via ModelScope）
API_EMBEDDING_BASE_URL=https://api-inference.modelscope.cn/v1
API_EMBEDDING_API_KEY=ms-xxx
API_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B

# 网络搜索
TAVILY_API_KEY=tvly-xxx
```

### 2. Python 依赖

本地开发需要完整依赖（含 OCR/解析链）：

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

仅跑 API 服务（或构建容器）用精简集合：

```bash
pip install -r requirements-api.txt
```

---

## 操作手册

### A. 启动 Milvus（向量库底座）

项目本身不管理 Milvus 容器，使用官方 standalone：

```bash
# 若还没下载过 milvus standalone，从官方拿 docker-compose 再 up
curl https://github.com/milvus-io/milvus/releases/download/v2.6.0/milvus-standalone-docker-compose.yml -o milvus-docker-compose.yml
docker compose -f milvus-docker-compose.yml up -d
```

默认监听 `19530`。Attu 可以连 `http://localhost:9091` 观察 collection。

### B. 创建知识库（首次初始化）

把 PDF 丢进 `datas/pdf/`（或把 Markdown 丢进 `datas/md/`），运行：

```bash
# 首次构建（destructive，清空已有 collection 并重建 schema/索引）
python -m documents.write_pdf_milvus --rebuild
# 或 Markdown 源
python -m documents.write_milvus --rebuild
```

执行过程：
1. 扫描目录下所有 `.pdf` / `.md`
2. 解析进程：逐页 OCR（或 Markdown 元素解析）→ 标题合并 → 语义切分
3. 写入进程：从队列接批次，调用 `add_documents` 批量写 Milvus

首次对单篇学术 PDF 的 OCR 耗时约 **1–3 分钟/10 页**（取决于 glm-ocr 服务器负载）。OCR 结果会写入 PDF 同名的 `.md` 文件作为缓存。

### C. 增量添加知识库（日常灌库）

不加 `--rebuild` 参数即为**增量模式**：

```bash
python -m documents.write_pdf_milvus          # 只处理 datas/pdf 下没入库过的
python -m documents.write_milvus              # 只处理 datas/md 下没入库过的
python -m documents.write_pdf_milvus --dir /other/path  # 指定其他目录
```

去重依据是 Milvus 里已存在记录的 `filename` 字段，单文件粒度。如果你想替换某篇 PDF 的内容，当前做法是先在 Milvus 删掉那个 filename 对应的记录，再跑增量。

### D. 命令行调试（不启服务）

```bash
python -m graph.graph1       # Corrective RAG
python -m graph2.graph_2     # Adaptive RAG（主力）
```

输入问题回车，图结构会 `graph.stream` 出每个节点的产物。输入 `q` 退出。

### E. Gradio 调试 UI

```bash
python debug_ui.py
```

默认监听 `http://localhost:7860`，聊天界面背后跑的就是 graph2。适合做路由与检索链路的直观排查。

### F. FastAPI 本地运行（非容器）

```bash
NO_PROXY=localhost,127.0.0.1 venv/bin/python -m uvicorn api.server:app --host 127.0.0.1 --port 8765
```

`NO_PROXY` 是 macOS 下必要的环境变量：Clash/V2Ray 若在 `127.0.0.1:7890` 监听，会劫持 Python httpx 对 localhost 的请求；`api/server.py` 顶部也会做 `setdefault`，但入口命令行显式声明更稳。

端点验证：

```bash
curl http://127.0.0.1:8765/health
# {"status":"ok"}

curl -X POST http://127.0.0.1:8765/query \
  -H "Content-Type: application/json" \
  -d '{"question":"铁死亡通过什么机制抑制三阴性乳腺癌？"}'
```

返回结构：

```json
{
  "answer": "...",
  "route": ["retrieve","grade_documents","decide_to_generate","generate"],
  "documents": [{"content": "...", "metadata": {"source":"02.pdf", "category":"content", ...}}],
  "error": null
}
```

### G. Docker 部署

```bash
# 构建并启动容器（宿主 Milvus 需已在 19530 运行）
docker compose up --build -d

# 健康检查
curl http://127.0.0.1:8765/health

# 查日志
docker logs -f rag-graph2-api

# 停止
docker compose down
```

要点：
- 容器内通过 `host.docker.internal:19530` 访问宿主 Milvus（`docker-compose.yml` 里的 `extra_hosts`）。
- `.env` 通过 `env_file` 注入容器；`MILVUS_URI` 在 compose 里显式覆盖为 `host.docker.internal`。
- `requirements-api.txt` 不含 `unstructured` / `pypdfium2`，所以**容器不负责入库**，只跑查询服务。入库依然要在宿主上执行。

### H. 作为工具被上游 Agent 调用

上游 `ReActAgents` 项目通过下述 LangChain 工具调用本服务（见该项目的 `backend/app/core/tools.py`）：

```python
@tool(
    "query_knowledge_base",
    args_schema=QueryKnowledgeBaseArgs,
    response_format="content_and_artifact",
)
async def query_knowledge_base(question: str) -> tuple[str, dict]:
    resp = await rag_client.post("/query", json={"question": question})
    ...
    # content: "KB_OK: ..." / "KB_EMPTY" / "KB_ERROR: timeout"
    # artifact: {"route": [...], "sources": [...], ...}
```

- `content` 字段是 LLM 看到的部分，只有简短状态串，避免 token 爆炸。
- `artifact` 字段留给 UI / 日志 / 评估，不回到 LLM 上下文。

---

## 已知限制与踩坑记录

项目跑通路上积累的几个必须绕过的坑，详细版本在 `CLAUDE.md`：

| 现象 | 解决 |
|---|---|
| `httpx` 连内网 Ollama 全部返回 502 | 强制 IPv4（`HTTPTransport(local_address="0.0.0.0")`）；OCR 路径直接 `subprocess.run(["curl", ...])` 绕开 Python HTTP 层 |
| `glm-ocr:bf16` 加载失败 | 请求 payload 必须带 `"options": {"num_ctx": 16384}` |
| `langchain-milvus 0.3.3` + `pymilvus 2.6.x` `_extract_fields` 报 alias 未注册 | `milvus_db.py::create_connection` 做 monkey-patch，首次访问前补注册 |
| macOS Clash 劫持 localhost | 所有入口脚本 `os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")` |
| glm-5 gateway 不支持 `response_format=json_schema` | 所有 `with_structured_output(...)` 传 `method="function_calling"` |
| Ollama 拒绝 token 数组输入 | OpenAIEmbeddings 设 `check_embedding_ctx_length=False` |

---

## 开发约定

- 所有注释与日志字符串使用中文；日志统一走 `utils.log_utils.log`（loguru）。
- Milvus schema 中文本字段最大 6000 字符，超出部分由 `SemanticChunker` 切；jieba 分词 + `cnalphanumonly` filter。
- Document metadata 中 `category` 用来区分 `Title` / `content` / `NarrativeText`；检索端只取 `content`，保证返给 LLM 的是合并后的标题-内容块，而不是孤立标题。
- 新增节点或 chain 优先放进 `graph2/`，graph1 作为对比/演示保留。
