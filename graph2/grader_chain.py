from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_models.api_llm import model


class GradeDocument(BaseModel):
    """对检索到的文档进行相关性评分的二元判断"""

    binary_score: str = Field(
        ...,
        description="判断文档是否与问题相关，取值为'yes'或'no'",
    )


# 带函数调用的LLM初始化
structured_llm_grader = model.with_structured_output(GradeDocument, method="function_calling")

system = """你是一个专业的文档相关性判断器。
你的任务是判断给定的文档是否与用户的问题相关。
如果文档与问题相关，输出'yes'；如果文档与问题不相关，输出'no'。"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统提示词
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ]
)

retrieval_grader_chain = grade_prompt | structured_llm_grader
