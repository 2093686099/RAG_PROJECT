from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_models.api_llm import model
from utils.log_utils import log


def transform_query(state):
    """
    优化检索查询，生成更适合向量数据库召回的英文问题

    Args:
        state (dict): 当前图状态，包含原始问题、检索查询和检索结果

    Returns:
        state (dict): 更新后的状态，仅替换 retrieval_query
    """
    log.info("---TRANSFORM QUERY---")  # 阶段标识
    question = state["question"]  # 获取原始问题
    retrieval_query = state.get("retrieval_query", question)  # 获取当前检索查询
    documents = state["documents"]  # 获取当前文档

    transform_count = state.get("transform_count", 0)  # 获取转换次数，默认0

    system = """作为生物医学检索查询重写器，您需要将输入问题转换为更适合英文向量数据库检索的优化版本。\n
             请结合原始用户问题理解其真实意图，保留基因名、数据集编号、蛋白名等专有名词原文。\n
             只输出一个更适合检索的英文问题，不要添加解释。"""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),  # 系统角色设定：问题优化器
            (
                "human",  # 用户输入模板
                "原始用户问题: \n\n {question} \n\n 当前检索查询: \n\n {retrieval_query} \n\n 请生成一个优化后的英文检索问题。",
            ),
        ]
    )

    # 构建问题重写处理链
    question_rewriter = (
            re_write_prompt  # 使用优化提示模板
            | model  # 调用语言模型
            | StrOutputParser()  # 将输出解析为字符串
    )

    # 问题重写
    better_query = question_rewriter.invoke({
        "question": question,
        "retrieval_query": retrieval_query,
    })  # 调用问题重写器
    return {
        "documents": documents,
        "question": question,
        "retrieval_query": better_query,
        "transform_count": transform_count + 1,
    }  # 返回更新状态
