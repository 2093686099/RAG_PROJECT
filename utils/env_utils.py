import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "t_collection01")

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# API 侧 LLM
API_LLM_BASE_URL = os.getenv("API_LLM_BASE_URL")
API_LLM_API_KEY = os.getenv("API_LLM_API_KEY")
API_LLM_MODEL = os.getenv("API_LLM_MODEL")

# API 侧 Embedding
API_EMBEDDING_BASE_URL = os.getenv("API_EMBEDDING_BASE_URL")
API_EMBEDDING_API_KEY = os.getenv("API_EMBEDDING_API_KEY")
API_EMBEDDING_MODEL = os.getenv("API_EMBEDDING_MODEL")
