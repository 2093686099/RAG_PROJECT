from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch

from utils.env_utils import (
    API_EMBEDDING_API_KEY,
    API_EMBEDDING_BASE_URL,
    API_EMBEDDING_MODEL,
    API_LLM_API_KEY,
    API_LLM_BASE_URL,
    API_LLM_MODEL,
)

model = ChatOpenAI(
    base_url=API_LLM_BASE_URL,
    api_key=API_LLM_API_KEY,
    model=API_LLM_MODEL,
    temperature=0,
    timeout=30,
    max_retries=2,
)

api_embeddings = OpenAIEmbeddings(
    model=API_EMBEDDING_MODEL,
    openai_api_base=API_EMBEDDING_BASE_URL,
    openai_api_key=API_EMBEDDING_API_KEY,
    check_embedding_ctx_length=False,
)

web_search_tool = TavilySearch(max_results=3)
