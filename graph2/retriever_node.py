from langchain_core.messages import HumanMessage, SystemMessage

from llm_models.api_llm import model
from tools.retriever_tools import retriever
from utils.log_utils import log

_TRANSLATE_SYSTEM = (
    "你是生物医学翻译专家。将用户问题翻译成英文用于英文文献检索。"
    "保留基因名、数据集编号、蛋白名等专有名词原文。只输出译文，不加任何解释。"
    "若问题已是英文则原样返回。"
)


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

    # 将中文问题译为英文以匹配英文语料库
    search_query = model.invoke([
        SystemMessage(content=_TRANSLATE_SYSTEM),
        HumanMessage(content=question),
    ]).content.strip()
    log.info(f"---检索查询（英文）: {search_query}---")

    documents = retriever.invoke(search_query)
    return {"documents": documents, "question": question}
