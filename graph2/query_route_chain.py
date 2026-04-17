from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_models.api_llm import model


# 查询的动态路由：根据用户的提问，决策采用哪种检索策略（网络检索，RAG）

class RouteQuery(BaseModel):
    """将用户查询查询路由到最相关的数据源"""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="根据用户问题选择将其路由到向量知识库或网络搜索",
    )


structured_llm_router = model.with_structured_output(RouteQuery, method="function_calling")

# 提示词
system = """你是一个擅长将用户问题路由到向量知识库或网络搜索的专家。
向量知识库包含与以下主题相关的生物医学文献：
- 铁死亡（ferroptosis）的机制、调控通路、脂质代谢、与疾病（尤其是癌症）的关系
- 三阴性乳腺癌（TNBC）相关的分子机制、靶点与治疗研究
- 基于 GSE58135 等 TCGA/GEO 公共数据集的转录组分析
- LASSO 回归构建多基因预后模型、风险评分与基因签名
- 铁死亡相关蛋白靶点的结构分析与药物研究

当用户问题涉及上述主题时，请路由到向量知识库（vectorstore）。
当问题是时事、实时信息、与上述主题无关的通用知识时，请路由到网络搜索（web_search）。"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统提示词
        ("human", "{question}"),  # 用户问题占位符
    ]
)

question_router_chain = route_prompt | structured_llm_router

# # 测试路由器
# print(
#     question_router_chain.invoke(
#         {"question": "什么是EUV光刻技术?"}
#     )
# )
# print(
#     question_router_chain.invoke(
#         {"question": "今天，长沙的天气怎么样?"}
#     )
# )
