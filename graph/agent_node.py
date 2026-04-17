# agent节点
from graph.graph_state1 import AgentState
from llm_models.all_llm import llm
from tools.retriever_tools import retriever_tool
from utils.log_utils import log


def agent_node(state: AgentState):
    """
    调用智能体模型基于当前状态生成响应。根据问题，
    它会决定使用检索工具检索，或直接结束。

    Args:
        state (messages): 当前状态

    Returns:
        dict: 更新后的状态，包含模型响应的消息对象
    """
    log.info("---调用智能体---")
    messages = state["messages"]

    model = llm.bind_tools([retriever_tool])
    response = model.invoke([messages[-1]])
    # 返回列表，因为这会添加到现有列表中
    return {"messages": [response]}