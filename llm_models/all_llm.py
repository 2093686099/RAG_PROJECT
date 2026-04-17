import httpx
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from utils.env_utils import QWEN_BASE_URL

# llama.cpp 与 httpx 默认连接方式不兼容，需强制 IPv4
_http_client = httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0"))

model = ChatOpenAI(
    model_name="qwen3.5",
    base_url="http://127.0.0.1:8088/v1",
    api_key="not-needed",
    temperature=0.5,
    http_client=_http_client,
)

llm = ChatOpenAI(
    model_name="qwen3.5",
    base_url="http://127.0.0.1:8088/v1",
    api_key="not-needed",
    temperature=0.5,
    http_client=_http_client,
)

web_search_tool = TavilySearch(max_results=3)

if __name__ == "__main__":
    agent = create_agent(
        model=model,
        tools=[web_search_tool]
    )

    res = agent.invoke(
        input={"messages": [{"role": "user", "content": "今天深圳天气"}]},
    )

    print(res["messages"][-1].content)

    # res = agent.invoke(
    #     {"messages": [{"role": "user", "content": "你好"}]}
    # )
    # print(res["messages"][-1].content)
