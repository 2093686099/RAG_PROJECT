import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from fastapi import FastAPI, HTTPException

from api.schemas import DocumentItem, HealthResponse, QueryRequest, QueryResponse
from graph2.graph_2 import graph
from utils.log_utils import log

app = FastAPI(
    title="RAG Graph2 Service",
    description="Adaptive RAG 知识库查询服务，供 ReActAgents 调用",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    route: list[str] = []
    final_answer = ""
    final_documents: list = []

    try:
        for output in graph.stream({"question": question}):
            for node, value in output.items():
                route.append(node)
                if not isinstance(value, dict):
                    continue
                if value.get("generation"):
                    final_answer = value["generation"]
                if value.get("documents"):
                    final_documents = value["documents"]
    except Exception as e:
        log.exception(e)
        return QueryResponse(
            answer="",
            route=route,
            documents=[],
            error=f"{type(e).__name__}: {e}",
        )

    docs_payload = [
        DocumentItem(
            content=getattr(d, "page_content", str(d)),
            metadata=getattr(d, "metadata", {}) or {},
        )
        for d in final_documents
    ]

    return QueryResponse(
        answer=final_answer or "(未生成回答)",
        route=route,
        documents=docs_payload,
    )
