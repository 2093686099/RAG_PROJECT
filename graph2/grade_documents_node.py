from graph2.grader_chain import retrieval_grader_chain
from utils.log_utils import log


def grade_documents(state):
    """
    评估检索到的文档与问题的相关性

    Args:
        state (dict): 当前图状态，包含问题和检索结果

    Returns:
        state (dict): 更新后的状态，documents字段仅保留相关文档
    """
    log.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")  # 打印当前阶段标识
    question = state["question"]  # 获取用户问题
    documents = state["documents"]  # 获取待评估文档

    # 文档评分与过滤
    filtered_docs = []
    for d in documents:  # 遍历所有文档
        score = retrieval_grader_chain.invoke(
            {"question": question, "document": d.page_content}
        )
        if score is None:
            # glm-5 偶发不发 tool call，function_calling parser 返回 None；保守丢弃
            log.warning("---GRADE: 评分器返回 None（无 tool call），按不相关丢弃---")
            continue
        grade = score.binary_score  # 获取二元评分结果
        if grade == "yes":
            log.info("---GRADE: 打印相关标识---")  # 打印相关标识
            filtered_docs.append(d)  # 添加到相关文档列表
        else:  # 如果文档不相关
            log.info("---GRADE: 打印不相关标识,并丢掉doc---")  # 打印不相关标识
            continue  # 跳过当前文档
    return {"documents": filtered_docs, "question": question}  # 返回仅含相关文档的状态
