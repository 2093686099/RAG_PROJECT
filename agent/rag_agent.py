from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from llm_models.all_llm import model
from tools.retriever_tools import retriever_tool

agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt='你是一个专业的半导体和芯片相关问答机器人，尽可能的调用工具回答用户的问题。如果工具中无法回答问题，请回答“我无法回答这个问题”。',
    checkpointer=InMemorySaver(),
)

if __name__ == '__main__':
    resp1 = agent.invoke(
        {"messages": [{"role": "user", "content": "什么是EUV光刻机？"}]},
        {"configurable": {"thread_id": "1"}},
    )

    print(resp1['messages'])
