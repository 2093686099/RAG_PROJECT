from langchain_core.messages import HumanMessage, SystemMessage

from llm_models.api_llm import model
from utils.log_utils import log

_TRANSLATE_SYSTEM = (
    "你是生物医学翻译专家。将用户问题翻译成英文用于英文文献检索。"
    "保留基因名、数据集编号、蛋白名等专有名词原文。只输出译文，不加任何解释。"
    "若问题已是英文则原样返回。"
)


def _looks_like_english(text: str) -> bool:
    """粗略判断输入是否已是英文/ASCII 查询，避免每次都调用翻译模型。"""
    return text.isascii()


def prepare_retrieval_query(state):
    """
    准备向量库检索用查询语句。

    Args:
        state (dict): 当前图状态，包含原始用户问题

    Returns:
        state (dict): 新增 retrieval_query 字段，保留原始 question 不变
    """
    question = state["question"]

    if _looks_like_english(question):
        log.info("---原问题已是英文/ASCII，直接用于检索---")
        return {"retrieval_query": question}

    log.info("---将用户问题翻译为英文检索查询---")
    try:
        retrieval_query = model.invoke([
            SystemMessage(content=_TRANSLATE_SYSTEM),
            HumanMessage(content=question),
        ]).content.strip()
    except Exception as exc:
        log.warning(f"---检索查询翻译失败，回退原问题: {exc}---")
        retrieval_query = question

    if not retrieval_query:
        log.warning("---检索查询翻译结果为空，回退原问题---")
        retrieval_query = question

    log.info(f"---检索查询: {retrieval_query}---")
    return {"retrieval_query": retrieval_query}
