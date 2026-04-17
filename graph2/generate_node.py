from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from graph2.graph_state import GraphState
from llm_models.api_llm import model


def generate(state: GraphState):
    """
    生成答案。

    Args:
        state (dict): 当前状态，包含问题和检索结果

    Returns:
        state (dict): 更新后的状态，新增包含生成结果的generation字段
    """
    question = state["question"]
    documents = state["documents"]

    # 提示模板
    prompt = PromptTemplate(
        template="你是一个问答任务助手，请根据以下检索到的上下文内容回答问题。如果不知道答案，请直接说明。回答保持简洁。\n问题：{question}\n上下文：{context} \n",
        input_variables=["question", "context"]
    )

    def format_docs(docs):
        """将多个文档内容合并为一个字符串，用两个换行符分隔每个文档"""
        if isinstance(docs, list):
            return "\n\n".join(doc.page_content for doc in docs)
        else:
            return "\n\n" + docs.page_content

    # 处理链
    rag_chain = prompt | model | StrOutputParser()

    # RAG 生成过程
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    return {
        "documents": documents,
        "generation": generation,
        "question": question,
        "generate_retry_count": state.get("generate_retry_count", 0) + 1,
    }
