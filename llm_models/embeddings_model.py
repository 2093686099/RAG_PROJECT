import httpx
from langchain_openai import OpenAIEmbeddings

# httpx 与 Ollama 服务端不兼容，需强制 IPv4
_http_client = httpx.Client(
    transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    trust_env=False,
)

ollama_embeddings = OpenAIEmbeddings(
    model="dengcao/Qwen3-Embedding-8B:Q4_K_M",
    openai_api_key="not-needed",
    openai_api_base="http://localhost:11434/v1",
    http_client=_http_client,
    check_embedding_ctx_length=False,
)