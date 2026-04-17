from typing import Literal, List

import uuid
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from graph.agent_node import agent_node
from graph.draw_png import draw_graph
from graph.generate_node import generate
from graph.get_human_message import get_last_human_message
from graph.graph_state1 import AgentState, Grade
from graph.rewrite_node import rewrite
from llm_models.all_llm import llm
from tools.retriever_tools import retriever_tool
from utils.log_utils import log
from utils.print_utils import _print_event


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    判断检索到的文档是否与问题相关
    Args:
        state (messages): 当前状态
    Returns:
        str: 判断结果，文档是否相关
    """
    log.info("---检查document的相关性---")
    # 带结构化输出的LLM
    llm_with_structured = llm.with_structured_output(Grade)

    # 提示模板
    prompt = PromptTemplate(
        template="""你是一个评估检索文档与用户问题相关性的评分器。
                这是检索到的文档：

                {context} 

                这是用户的问题：{question} 

                任务：仅基于文档内容判断是否包含与用户问题相关的信息。
                输出格式：严格只返回JSON格式，不包含任何其他解释或说明。
                输出内容：{{"binary_score": "yes"}} 或 {{"binary_score": "no"}}。

                示例输出：{{"binary_score": "yes"}}
                """,
        input_variables=["context", "question"]
    )

    # 处理链c
    chain = prompt | llm_with_structured

    messages = state["messages"]
    question = get_last_human_message(messages).content
    last_message = messages[-1]
    docs = last_message.content

    # 执行
    scored_result = chain.invoke({"context": docs, "question": question})

    score = scored_result.binary_score

    if score == "yes":
        log.info("---文档相关---")
        return "generate"
    else:
        log.info("---文档不相关---")
        return "rewrite"


# 定义一个新的工作流图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node('agent', agent_node)
workflow.add_node('retrieve', ToolNode([retriever_tool]))
workflow.add_node('rewrite', rewrite)
workflow.add_node('generate', generate)

workflow.add_edge(START, 'agent')
workflow.add_conditional_edges(
    'agent',
    tools_condition,
    {
        'tools': 'retrieve',
        END: END
    }
)

workflow.add_conditional_edges(
    'retrieve',
    grade_documents,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    }
)

workflow.add_edge('rewrite', 'agent')
workflow.add_edge('generate', END)

memory = MemorySaver()

graph = workflow.compile(checkpointer=memory)

# draw_graph(graph, 'graph_rag1-2.png')

config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
    }
}

_printed = set()  # set集合，避免重复打印

# 执行工作流
while True:
    question = input('用户：')
    if question.lower() in ['q', 'quit', 'exit']:
        log.info('对话结束，拜拜！')
        break
    else:
        inputs = {
            "messages": [
                ("user", question),
            ]
        }
        events = graph.stream(inputs, config=config, stream_mode='values')
        # 打印消息
        for event in events:
            _print_event(event, _printed)