from typing import TypedDict, List

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    表示图处理流程的状态信息

    属性说明：
        question: 用户提出的问题文本
        retrieval_query: 用于向量库检索的查询语句
        transform_count:  转换查询的次数
        generation: 语言模型生成的回答文本
        documents: 检索到的相关文档列表
    """

    question: str
    retrieval_query: str
    transform_count: int  # 转换查询的次数
    generate_retry_count: int  # generate 节点因幻觉被判不支持的重试次数
    generation: str
    documents: List[Document]
