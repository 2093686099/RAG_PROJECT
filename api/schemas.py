from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    """检索到的文档片段"""
    content: str
    metadata: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(..., min_length=1, description="用户问题")


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str = Field(..., description="生成的回答")
    route: List[str] = Field(default_factory=list, description="graph 执行路径")
    documents: List[DocumentItem] = Field(default_factory=list, description="检索到的文档")
    error: Optional[str] = Field(default=None, description="错误信息，若有")


class HealthResponse(BaseModel):
    status: str = "ok"
