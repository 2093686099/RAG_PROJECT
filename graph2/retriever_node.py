from tools.retriever_tools import retriever
from utils.log_utils import log


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
    return {"documents": documents, "question": question, "retrieval_query": retrieval_query}
