from langchain_core.tools import create_retriever_tool

from documents.milvus_db import MilvusVectorSave
from llm_models.api_llm import api_embeddings

mv = MilvusVectorSave(embedding_function=api_embeddings)
mv.create_connection()
retriever = mv.vector_store_saved.as_retriever(
    search_type='similarity',
    search_kwargs={
        "k": 5,
        "ranker_type": "rrf",
        "ranker_params": {"k": 100},
        "score_threshold": 0.1,
        'filter': {"category": "content"}
    }
)

retriever_tool = create_retriever_tool(
    retriever,
    'rag_retriever',
    '搜索并返回关于‘铁死亡’的内容，内容涵盖：铁死亡的机制，铁死亡的检测，铁死亡的药物，铁死亡的临床应用等'
)
