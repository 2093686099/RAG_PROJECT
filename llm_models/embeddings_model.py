import httpx
from langchain_openai import OpenAIEmbeddings

# httpx 与 Ollama 服务端不兼容，需强制 IPv4
_http_client = httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0"))

ollama_embeddings = OpenAIEmbeddings(
    model="qwen3-embedding:8b",
    openai_api_key="not-needed",
    openai_api_base="http://100.112.63.27:11434/v1",
    http_client=_http_client,
    check_embedding_ctx_length=False,
)