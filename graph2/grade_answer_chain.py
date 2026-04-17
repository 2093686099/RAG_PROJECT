from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_models.api_llm import model


class GradeAnswer(BaseModel):
    """评估回答是否解决用户问题的二元评分模型"""

    binary_score: str = Field(
        description="用户问题是否被回答解决，'yes'表示解决，'no'表示未解决"
    )

structured_llm_grader = model.with_structured_output(GradeAnswer, method="function_calling")

# 提示词
system = """您是一个评估回答是否解决用户问题的评分器。\n
    给出一个二元评分，'yes'表示回答确实解决了用户问题，'no'表示未解决"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "用户问题：\n\n {question} \n\n 回答：{generation}"),
    ]
)

# 构建回答评估工作流
answer_grader_chain = answer_prompt | structured_llm_grader
