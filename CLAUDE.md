# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) knowledge base system, built with LangChain, LangGraph, and Milvus vector database. Currently ingests academic PDFs (biomedical domain) via OCR and serves them through agent and self-correcting LangGraph workflows.

## Running the Application

```bash
# Incremental-ingest Markdown files into Milvus (multi-process)
python -m documents.write_milvus
python -m documents.write_milvus --rebuild   # drop and recreate the collection

# Incremental-ingest PDF files into Milvus (multi-process, OCR-based)
python -m documents.write_pdf_milvus
python -m documents.write_pdf_milvus --rebuild

# Run the CLI-based Corrective RAG graph (graph1 — agent-style with rewrite loop)
python -m graph.graph1

# Run the CLI-based Adaptive RAG graph (graph2 — route/grade/hallucination checks)
python -m graph2.graph_2

# Export the graph2 PNG manually when needed
python -c "from graph2.graph_2 import export_graph_png; export_graph_png()"

# Launch the Gradio debug UI (chat against graph2 locally)
python debug_ui.py

# Launch the FastAPI service (serves graph2 as HTTP)
NO_PROXY=localhost,127.0.0.1 venv/bin/python -m uvicorn api.server:app --host 127.0.0.1 --port 8765

# Dockerized API
docker compose up --build
```

Ingestion is incremental by default: `ensure_collection()` creates the collection if absent, and `get_existing_filenames()` skips files already loaded. Use `--rebuild` to drop and repopulate.

## Environment

- `.env` provides:
  - Legacy/internal: `OPENAI_API_KEY`, `OPENAI_API_BASE`, `QWEN_BASE_URL`, `TAVILY_API_KEY`.
  - External API stack (used by `llm_models/api_llm.py` and the FastAPI service): `API_LLM_BASE_URL` / `API_LLM_API_KEY` / `API_LLM_MODEL` (glm-5 via tokenhub.tencentmaas.com), `API_EMBEDDING_BASE_URL` / `API_EMBEDDING_API_KEY` / `API_EMBEDDING_MODEL` (Qwen3-Embedding-8B via api-inference.modelscope.cn).
- `utils/env_utils.py` reads `MILVUS_URI` (default `http://localhost:19530`) and `COLLECTION_NAME` (default `t_collection01`) from env, so `docker-compose.yml` overrides `MILVUS_URI=http://host.docker.internal:19530` to reach the host-local Milvus from inside the container.
- Ollama embedding endpoint is in `llm_models/embeddings_model.py` (`http://localhost:11434`, model `dengcao/Qwen3-Embedding-8B:Q4_K_M`) — used only by ingestion.
- Internal chat LLM endpoint is in `llm_models/all_llm.py`: `http://127.0.0.1:8088/v1` (Cursor-forwarded tunnel to an internal llama.cpp serving `qwen3.5`) — not used by the API path.
- Key framework versions: `langchain-core 1.2.x`, `langchain-milvus 0.3.3`, `pymilvus 2.6.x`, `langgraph 1.1.x`.

## Architecture

### Core Layers

**LLM & Embeddings** (`llm_models/`)
- `all_llm.py` — legacy/internal: exposes `model` and `llm` (both `ChatOpenAI` pointing at the llama.cpp tunnel). Also creates `web_search_tool` (TavilySearch). Used by graph1 / legacy ingestion.
- `api_llm.py` — external API stack: exposes `model` (glm-5 via tokenhub.tencentmaas.com), `api_embeddings` (Qwen3-Embedding-8B via modelscope, dim=4096 — matches Milvus schema), and `web_search_tool`. All graph2 chains + the FastAPI service import from here.
- `embeddings_model.py` — exposes `ollama_embeddings` (`dengcao/Qwen3-Embedding-8B:Q4_K_M` via local Ollama on `localhost:11434`, OpenAI-compatible `/v1/embeddings` endpoint). Used by ingestion only.

**Document Parsing & Ingestion** (`documents/`)
- `markdown_parser.py` — parses Markdown with `UnstructuredMarkdownLoader` in `elements` mode, merges title+content via `merge_title_content`, then applies `SemanticChunker` for chunks > 8000 chars.
- `pdf_parser.py` — OCR pipeline: `pypdfium2` renders each page → local `dhiltgen/glm-ocr:bf16` on `localhost:11434` (via `curl --noproxy localhost` subprocess) produces Markdown → writes to a same-name `.md` next to the PDF (acts as OCR cache) → reuses `MarkdownParser` for chunking.
- `milvus_db.py` — `MilvusVectorSave` manages a Milvus collection with dense (HNSW, dim=4096) + sparse BM25 (standard tokenizer + lowercase + English stopwords) vectors. Key methods: `create_collection()` (destructive rebuild), `ensure_collection()` (non-destructive), `create_connection()`, `add_documents()`, `get_existing_filenames()`.
- `write_milvus.py` / `write_pdf_milvus.py` — multi-process producer-consumer pipelines with `--rebuild` / `--dir` CLI flags.

**Retrieval Tools** (`tools/`)
- `retriever_tools.py` — LangChain retriever tool wrapping the Milvus vector store with RRF hybrid search (k=5, dense similarity + BM25).

**Agent** (`agent/`)
- `rag_agent.py` — a LangChain `create_agent` with the Milvus retriever tool and `InMemorySaver` checkpoint.

**LangGraph Workflows** — two self-correcting RAG pipelines:
- `graph/` (Graph 1 — Corrective RAG): `agent_node` → tool retrieval → `grade_documents` → branch to `generate` or `rewrite` → END. Uses message-based `AgentState`.
- `graph2/` (Graph 2 — Adaptive RAG): `route_question` (vectorstore vs web_search) → `prepare_retrieval_query` → `retrieve` → `grade_documents` → `decide_to_generate` → `generate` → hallucination check + answer grading → loop or END. Uses `GraphState` with `question` / `retrieval_query` / `documents` / `generation` / `generate_retry_count` / `transform_count`. `prepare_retrieval_query` generates an English-oriented retrieval query when useful but keeps the original `question` unchanged for routing, generation, and answer grading. `retrieve` does dual recall when `retrieval_query != question`: first the translated/optimized query, then the original question, followed by deduplication. `transform_query` rewrites `retrieval_query` rather than overwriting `question`. `generate_retry_count` caps hallucination retries at `MAX_GENERATE_RETRIES = 2` and falls back to `useful` to prevent infinite loops. All grader chains use `with_structured_output(..., method="function_calling")` because glm-5 doesn't support `json_schema` response format.

**HTTP Service** (`api/`)
- `api/schemas.py` — Pydantic models: `QueryRequest` (question), `QueryResponse` (answer + route + documents + error), `HealthResponse`.
- `api/server.py` — FastAPI app. `GET /health` for liveness, `POST /query` streams graph2 to collect the full route and final `generation` + `documents` into a blocking JSON response. Sets `NO_PROXY=localhost,127.0.0.1` at import to bypass the macOS Clash/V2Ray proxy on 127.0.0.1:7890.

**Debug UI**
- `debug_ui.py` — Gradio `ChatInterface` wrapping graph2 with streaming trace. Same proxy workaround. Launch on port 7860.

**Docker**
- `Dockerfile` (python:3.12-slim + `requirements-api.txt`) — copies only API-path modules (`api`, `graph`, `graph2`, `llm_models`, `tools`, `utils`). Exposes port 8000.
- `docker-compose.yml` — publishes `8765:8000`, mounts `.env` via `env_file`, injects `MILVUS_URI=http://host.docker.internal:19530` so the container reaches host-local Milvus.
- `requirements-api.txt` — API-only subset (fastapi, langchain, langchain-openai, langchain-milvus, langgraph, pymilvus, etc.). No `unstructured`, `pypdfium2`, HuggingFace/torch.

### Data Flow

1. PDFs → OCR (`dhiltgen/glm-ocr:bf16` on local Ollama) → Markdown cache → element extraction → title-content merge → semantic chunking → Milvus (dense + sparse).
2. User query → agent / graph → Milvus hybrid retrieval (RRF) → (grading / rewrite loop) → generation → response.

## Known Workarounds

- **httpx + internal tunnel**: all Python HTTP libraries return 502 against the remote Ollama server. Both `all_llm.py` and `embeddings_model.py` force IPv4 via `httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0"))`. For OCR (`pdf_parser.py`) we bypass Python HTTP entirely and shell out to `curl`.
- **`glm-ocr:bf16` crash**: Ollama needs `"options": {"num_ctx": 16384}` in the request payload, otherwise the model fails to load.
- **`langchain-milvus 0.3.3` + `pymilvus 2.6.x` alias bug**: the `Milvus` wrapper's internal `MilvusClient` uses a random alias (`cm-xxx`), but `_extract_fields` accesses `Collection(using=alias)` via the ORM registry, which isn't populated. `milvus_db.py::create_connection()` monkey-patches `Milvus._extract_fields` to register the connection on first access.
- **Ollama embedding input type**: LangChain tokenizes inputs before sending, but Ollama rejects token arrays. `embeddings_model.py` sets `check_embedding_ctx_length=False`.
- **macOS system proxy hijacks localhost**: when Clash/V2Ray runs on `127.0.0.1:7890`, Python httpx reads `HTTPS_PROXY`/`HTTP_PROXY` from env and routes localhost through the proxy, breaking Gradio/uvicorn local HTTP. `debug_ui.py` and `api/server.py` both `os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")` before any import that could load httpx.
- **glm-5 structured output**: the upstream gateway rejects `response_format=json_schema`. All `with_structured_output(...)` calls in graph2 pass `method="function_calling"`.
- **graph2 PNG export is manual**: `graph2.graph_2` no longer auto-renders `graph_rag2.png` on import/run. Call `export_graph_png()` explicitly when the diagram needs refreshing.

## Important Conventions

- All code comments and log strings are in Chinese.
- Logging uses `loguru` via `utils/log_utils.py` — import as `from utils.log_utils import log`.
- The Milvus schema uses a `standard` tokenizer + `lowercase` + English stopword filter for BM25 (optimized for English biomedical corpus); text field max length is 6000 chars.
- Document metadata convention: `category` field distinguishes `Title`, `content`, `NarrativeText`, `UncategorizedText`; the `merge_title_content` pattern builds hierarchical title→content relationships using `parent_id`/`element_id`.
