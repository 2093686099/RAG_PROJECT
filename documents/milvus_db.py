from langchain_core.documents import Document
from typing import List, Optional
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, Function, connections
from pymilvus.client.types import MetricType, IndexType, DataType, FunctionType

from llm_models.embeddings_model import ollama_embeddings
from utils.env_utils import MILVUS_URI, COLLECTION_NAME


class MilvusVectorSave:
    """把新的document数据插入到数据库中"""

    def __init__(self, embedding_function=None):
        """自定义collection

        Args:
            embedding_function: 覆盖默认 embedding；默认使用内网 Ollama（ingestion 场景）。
                API 侧检索应传入 api_embeddings 以避免对内网的依赖。
        """
        self.vector_store_saved: Milvus = None
        self.embedding_function = embedding_function if embedding_function is not None else ollama_embeddings

    def _build_schema_and_index(self, client: MilvusClient):
        """构建 collection 的 schema 和索引参数"""
        schema = client.create_schema()
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=4096)

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25
        )
        schema.add_function(function=bm25_function)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.6,
                "bm25_b": 0.75,
            },
        )
        index_params.add_index(
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 64}
        )
        return schema, index_params

    def create_collection(self):
        """删除旧表并重建（破坏性操作，会清空所有数据）"""
        client = MilvusClient(uri=MILVUS_URI)
        schema, index_params = self._build_schema_and_index(client)

        if COLLECTION_NAME in client.list_collections():
            client.release_collection(collection_name=COLLECTION_NAME)
            client.drop_index(collection_name=COLLECTION_NAME, index_name="sparse_inverted_index")
            client.drop_index(collection_name=COLLECTION_NAME, index_name="dense_inverted_index")
            client.drop_collection(collection_name=COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def ensure_collection(self):
        """确保 collection 存在，已存在则跳过（非破坏性）"""
        client = MilvusClient(uri=MILVUS_URI)
        if COLLECTION_NAME in client.list_collections():
            return
        schema, index_params = self._build_schema_and_index(client)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def get_existing_filenames(self) -> set:
        """查询已入库的所有 filename，用于增量写入时去重"""
        client = MilvusClient(uri=MILVUS_URI)
        if COLLECTION_NAME not in client.list_collections():
            return set()
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter='',
            output_fields=['filename'],
            limit=10000,
        )
        return {row['filename'] for row in results}

    def create_connection(self):
        """创建一个Connection： milvus + langchain"""
        # langchain-milvus 0.3.3 + pymilvus 2.6.x 有 bug：
        # Milvus wrapper 内部用 MilvusClient 连接（生成随机 alias "cm-xxx"），
        # 但 _extract_fields 通过 Collection ORM 接口访问，后者要求
        # connections 注册表里有对应 alias 的连接。
        # 修法：patch _extract_fields，在首次访问 self.col 前把连接补注册上。
        _original_extract = Milvus._extract_fields

        def _patched_extract(self_milvus):
            if not connections.has_connection(self_milvus.alias):
                connections.connect(alias=self_milvus.alias, uri=MILVUS_URI)
            return _original_extract(self_milvus)

        Milvus._extract_fields = _patched_extract
        try:
            self.vector_store_saved = Milvus(
                embedding_function=self.embedding_function,
                collection_name=COLLECTION_NAME,
                builtin_function=BM25BuiltInFunction(),
                vector_field=['dense', 'sparse'],
                consistency_level="Strong",
                auto_id=True,
                connection_args={"uri": MILVUS_URI}
            )
        finally:
            Milvus._extract_fields = _original_extract

    def add_documents(self, datas: List[Document]):
        """添加新的document数据到Milvus"""
        self.vector_store_saved.add_documents(datas)


if __name__ == '__main__':
    from documents.markdown_parser import MarkdownParser
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(r'F:\PycharmProjects\RAG_PROJECT\test.md')

    # 写入Milvus数据库
    mv = MilvusVectorSave()
    mv.create_collection()
    mv.create_connection()
    mv.add_documents(docs)

    client = mv.vector_store_saved.client
    # 得到表结构
    desc_collection = client.describe_collection(
        collection_name=COLLECTION_NAME
    )
    print("表结构是:", desc_collection)

    # 得到索引信息
    res = client.list_indexes(
        collection_name=COLLECTION_NAME,
    )
    print("索引信息是:", res)

    if res:
        for i in res:
            # 得到的索引描述
            desc_index = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=i
            )
            print("索引描述是:", desc_index)

    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="category == 'Title'",  # 查询条件
        output_fields=['text', 'category', 'filename'],  # 返回的字段
    )
    print("查询结果是:", result)

    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="category == 'content'",  # 查询 category == 'content' 的所有数据
        output_fields=['text', 'category', 'filename']  # 指定返回的字段
    )

    print('测试 过滤查询的结果是: ', result)