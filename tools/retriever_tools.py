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
    '搜索并返回关于‘半导体和芯片’的内容，内容涵盖：半导体和芯片的封装测试，光刻胶等'
)
