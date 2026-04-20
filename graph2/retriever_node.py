from langchain_core.documents import Document

from tools.retriever_tools import retriever
from utils.log_utils import log


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    """按来源和内容去重，避免双路召回返回重复片段。"""
    deduped_documents = []
    seen = set()

    for document in documents:
        source = document.metadata.get("source", "")
        key = (source, document.page_content)
        if key in seen:
            continue
        seen.add(key)
        deduped_documents.append(document)

    return deduped_documents


def retrieve(state):
    """
    检索相关文档

    Args:
        state (dict): 当前图状态，包含用户问题

    Returns:
        state (dict): 更新后的图状态，新增包含检索到的文档列表
    """
    log.info("---去知识库中检索文档---")
    question = state["question"]
    retrieval_query = state.get("retrieval_query", question)
    log.info(f"---当前检索查询: {retrieval_query}---")

    documents = retriever.invoke(retrieval_query)
    if retrieval_query != question:
        log.info("---追加原始问题召回，合并中英双路结果---")
        documents.extend(retriever.invoke(question))
        documents = _dedupe_documents(documents)

    return {"documents": documents, "question": question, "retrieval_query": retrieval_query}
