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
```

Ingestion is incremental by default: `ensure_collection()` creates the collection if absent, and `get_existing_filenames()` skips files already loaded. Use `--rebuild` to drop and repopulate.

## Environment

- `.env` provides `OPENAI_API_KEY`, `OPENAI_API_BASE`, `QWEN_BASE_URL`, `TAVILY_API_KEY`.
- Hardcoded infra addresses live in `utils/env_utils.py`: Milvus URI (`MILVUS_URI = http://localhost:19530`), collection name (`COLLECTION_NAME = t_collection01`). Ollama embedding endpoint is in `llm_models/embeddings_model.py` (`http://100.112.63.27:11434`).
- Chat LLM endpoint is in `llm_models/all_llm.py`: `http://127.0.0.1:8088/v1` (Cursor-forwarded tunnel to an internal llama.cpp serving `qwen3.5`).
- Key framework versions: `langchain-core 1.2.x`, `langchain-milvus 0.3.3`, `pymilvus 2.6.x`, `langgraph 1.1.x`.

## Architecture

### Core Layers

**LLM & Embeddings** (`llm_models/`)
- `all_llm.py` — exposes `model` and `llm` (both `ChatOpenAI` pointing at the llama.cpp tunnel). Also creates `web_search_tool` (TavilySearch).
- `embeddings_model.py` — exposes `ollama_embeddings` (`qwen3-embedding:8b` via remote Ollama, accessed through the OpenAI-compatible `/v1/embeddings` endpoint).

**Document Parsing & Ingestion** (`documents/`)
- `markdown_parser.py` — parses Markdown with `UnstructuredMarkdownLoader` in `elements` mode, merges title+content via `merge_title_content`, then applies `SemanticChunker` for chunks > 8000 chars.
- `pdf_parser.py` — OCR pipeline: `pypdfium2` renders each page → remote `glm-ocr:bf16` (via `curl` subprocess) produces Markdown → writes to a same-name `.md` next to the PDF (acts as OCR cache) → reuses `MarkdownParser` for chunking.
- `milvus_db.py` — `MilvusVectorSave` manages a Milvus collection with dense (HNSW, dim=4096) + sparse BM25 (jieba tokenizer) vectors. Key methods: `create_collection()` (destructive rebuild), `ensure_collection()` (non-destructive), `create_connection()`, `add_documents()`, `get_existing_filenames()`.
- `write_milvus.py` / `write_pdf_milvus.py` — multi-process producer-consumer pipelines with `--rebuild` / `--dir` CLI flags.

**Retrieval Tools** (`tools/`)
- `retriever_tools.py` — LangChain retriever tool wrapping the Milvus vector store with RRF hybrid search (k=5, dense similarity + BM25).

**Agent** (`agent/`)
- `rag_agent.py` — a LangChain `create_agent` with the Milvus retriever tool and `InMemorySaver` checkpoint.

**LangGraph Workflows** — two self-correcting RAG pipelines:
- `graph/` (Graph 1 — Corrective RAG): `agent_node` → tool retrieval → `grade_documents` → branch to `generate` or `rewrite` → END. Uses message-based `AgentState`.
- `graph2/` (Graph 2 — Adaptive RAG): `route_question` (vectorstore vs web_search) → `retrieve` → `grade_documents` → `decide_to_generate` → `generate` → hallucination check + answer grading → loop or END. Uses `GraphState` with explicit `question`/`documents`/`generation` fields.

### Data Flow

1. PDFs → OCR (`glm-ocr:bf16`) → Markdown cache → element extraction → title-content merge → semantic chunking → Milvus (dense + sparse).
2. User query → agent / graph → Milvus hybrid retrieval (RRF) → (grading / rewrite loop) → generation → response.

## Known Workarounds

- **httpx + internal tunnel**: all Python HTTP libraries return 502 against the remote Ollama server. Both `all_llm.py` and `embeddings_model.py` force IPv4 via `httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0"))`. For OCR (`pdf_parser.py`) we bypass Python HTTP entirely and shell out to `curl`.
- **`glm-ocr:bf16` crash**: Ollama needs `"options": {"num_ctx": 16384}` in the request payload, otherwise the model fails to load.
- **`langchain-milvus 0.3.3` + `pymilvus 2.6.x` alias bug**: the `Milvus` wrapper's internal `MilvusClient` uses a random alias (`cm-xxx`), but `_extract_fields` accesses `Collection(using=alias)` via the ORM registry, which isn't populated. `milvus_db.py::create_connection()` monkey-patches `Milvus._extract_fields` to register the connection on first access.
- **Ollama embedding input type**: LangChain tokenizes inputs before sending, but Ollama rejects token arrays. `embeddings_model.py` sets `check_embedding_ctx_length=False`.

## Important Conventions

- All code comments and log strings are in Chinese.
- Logging uses `loguru` via `utils/log_utils.py` — import as `from utils.log_utils import log`.
- The Milvus schema uses a jieba tokenizer with `cnalphanumonly` filter for BM25; text field max length is 6000 chars.
- Document metadata convention: `category` field distinguishes `Title`, `content`, `NarrativeText`, `UncategorizedText`; the `merge_title_content` pattern builds hierarchical title→content relationships using `parent_id`/`element_id`.
