from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_models.api_llm import model


class GradeHallucinations(BaseModel):
    """对生成回答中是否存在幻觉进行二元评分"""

    binary_score: str = Field(
        description="回答是否基于事实，是则返回'yes'，否则返回'no'"
    )

structured_llm_grader = model.with_structured_output(GradeHallucinations, method="function_calling")

# 提示词
system = """您是一个评估生成内容是否基于检索事实的评分器。\n
    给出一个二元评分，'yes'表示生成内容是基于事实集，'no'表示存在幻觉"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "事实集：\n\n {documents} \n\n 生成内容：{generation}"),
    ]
)

# 构建幻觉检测工作流
hallucination_grader_chain = hallucination_prompt | structured_llm_grader
