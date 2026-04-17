from langchain_core.messages import HumanMessage

from graph.get_human_message import get_last_human_message
from graph.graph_state1 import AgentState
from llm_models.all_llm import llm
from utils.log_utils import log


def rewrite(state: AgentState):
    """
    转换查询以生成更好的问题。

    Args:
        state (messages): 当前状态，包含包含用户问题的消息对象

    Returns:
        dict: 更新后的状态，包含转换后的问题的消息对象
    """
    log.info("---转换查询---")
    messages = state["messages"]
    question = get_last_human_message(messages).content

    msg = [
        HumanMessage(
            content=f""" \n
    分析输入并尝试理解潜在的语义意图/含义。 \n
    这是初识问题：
    \n ------ \n
    {question}
    \n ------ \n
    请提出一个改进后的问题： """,
        )
    ]

    # 评分模型
    response = llm.invoke(msg)
    return {"messages": [response]}
