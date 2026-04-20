import uuid
from pprint import pprint

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from graph.draw_png import draw_graph
from graph2.generate_node import generate
from graph2.grade_answer_chain import answer_grader_chain
from graph2.grade_documents_node import grade_documents
from graph2.grade_hallucinations_chain import hallucination_grader_chain
from graph2.graph_state import GraphState
from graph2.prepare_retrieval_query_node import prepare_retrieval_query
from graph2.query_route_chain import question_router_chain
from graph2.retriever_node import retrieve
from graph2.transform_query_node import transform_query
from graph2.web_search_node import web_search
from utils.log_utils import log
from utils.print_utils import _print_event


def _format_docs_for_grading(documents):
    """将 documents 序列化为文本，兼容 list[Document] 和单个 Document"""
    if isinstance(documents, list):
        return "\n\n".join(doc.page_content for doc in documents)
    if documents is None:
        return ""
    return documents.page_content


MAX_GENERATE_RETRIES = 2
MAX_TRANSFORM_RETRIES = 1  # 问题重写后仍无相关文档，立即切 web_search 避免长循环烧 LLM


def grade_generation_v_documents_and_questiono(state):
    """
    评估生成结果是否基于文档并正确回答问题

    Args:
        state (dict): 当前图状态，包含问题、文档和生成结果

    Returns:
        str: 下一节点的名称（useful/not userful/not supported）
    """
    log.info("---检查生成内容是否存在幻觉---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state.get("generate_retry_count", 0)

    # 检查生成是否基于文档
    score = hallucination_grader_chain.invoke({
        "documents": _format_docs_for_grading(documents),
        "generation": generation
    })
    grade = score.binary_score

    if grade == "yes":
        log.info("---生成内容基于文档---")
        log.info("---评估生成回答与问题的匹配度---")
        score = answer_grader_chain.invoke({
            "question": question,
            "generation": generation
        })
        grade = score.binary_score
        if grade == "yes":
            log.info("---生成回答与问题匹配度高---")
            return "useful"
        log.info("---生成回答与问题匹配度低---")
        return "not useful"

    if retry_count >= MAX_GENERATE_RETRIES:
        log.warning(f"---幻觉重试已达上限 ({retry_count})，直接结束---")
        return "useful"
    log.info(f"---生成内容未基于参考文档，重试 {retry_count + 1}/{MAX_GENERATE_RETRIES}---")
    return "not supported"


def decide_to_generate(state):
    """
    决定是生成回答还是重新优化问题

    Args:
        state (dict): 当前图状态，包含问题和文档

    Returns:
        str: 下一个节点名称，"generate"或"transform_query"
    """
    log.info("---ASSESS GRADED DOCUMENTS---")

    filtered_documents = state["documents"]
    transform_count = state.get("transform_count", 0)  # 获取转换次数，默认0

    if not filtered_documents:  # 如果没有相关文档
        if transform_count >= MAX_TRANSFORM_RETRIES:
            log.info(f"---重写后仍无相关文档 (transform_count={transform_count})，切换 web_search---")
            return "web_search"
        log.info(f"---没有相关文档，重写问题 (transform_count={transform_count})---")
        return "transform_query"
    else:  # 如果有相关文档
        log.info("---有相关文档, 生成回答---")
        return "generate"


def route_question(state):
    """
    路由问题到网络搜索或RAG流程

    Args:
        state (dict): 当前图状态，包含用户问题

    Returns:
        str: 下一个节点名称，"web_search"或"retrieve"
    """
    log.info("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router_chain.invoke({"question": question})

    # 根据路由结果决定下一个节点
    if source.datasource == "web_search":
        log.info("---路由到网络搜索---")
        return "web_search"
    elif source.datasource == "vectorstore":
        log.info("---路由到文档检索---")
        return "retrieve"


workflow = StateGraph(GraphState)

# 定义个状态节点
workflow.add_node("web_search", web_search)  # 网络搜索节点
workflow.add_node("prepare_retrieval_query", prepare_retrieval_query)  # 检索查询准备节点
workflow.add_node("retrieve", retrieve)  # 文档检索节点
workflow.add_node("grade_documents", grade_documents)  # 文档评分节点
workflow.add_node("generate", generate)  # 回答生成节点
workflow.add_node("transform_query", transform_query)  # 查询优化节点

# 起始路由判断
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "retrieve": "prepare_retrieval_query",
    },
)

# 添加固定边
workflow.add_edge("web_search", "generate")
workflow.add_edge("prepare_retrieval_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# 文档评估后的条件分支
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate
)

# 生成结果评估后的条件分支
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_questiono,
    {
        "not supported": "generate",  # 生成不符合要求时重试
        "useful": END,  # 生成符合要求时结束
        "not useful": "transform_query",  # 生成无用结果时优化查询
    }
)

workflow.add_edge("transform_query", "retrieve")  # 查询优化后重新检索文档

graph = workflow.compile()

draw_graph(graph, 'graph_rag2.png')

if __name__ == '__main__':
    _printed = set()

    # 执行工作流
    while True:
        question = input('用户：')
        if question.lower() in ['q', 'quit', 'exit']:
            log.info('对话结束，拜拜！')
            break
        else:
            inputs = {
                "question": question
            }
            for output in graph.stream(inputs):
                for key, value in output.items():
                    pprint(f"Node '{key}':")
                    pprint(value)
                print("\n---\n")

            pprint(value["generation"])