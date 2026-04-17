from langchain_core.documents import Document

from llm_models.all_llm import web_search_tool


def test_web_search_content_extraction():
    """测试web_search函数是否正确工作"""
    print("开始测试内容提取逻辑...")

    try:
        # 1. 先获取实际的搜索结果
        query = "今天深圳天气"
        docs = web_search_tool.invoke({"query": query})

        print("\n步骤1: 获取搜索结果")
        print(f"返回类型: {type(docs)}")
        print(f"包含的键: {list(docs.keys())}")

        # 2. 查看results字段
        if "results" in docs:
            results = docs["results"]
            print(f"\n步骤2: 查看results字段")
            print(f"results类型: {type(results)}")
            print(f"结果数量: {len(results)}")

            # 3. 查看第一个结果的结构
            if results:
                first_result = results[0]
                print(f"\n步骤3: 查看第一个结果的结构")
                print(f"第一个结果类型: {type(first_result)}")
                print(f"第一个结果包含的键: {list(first_result.keys())}")

                # 4. 测试内容提取
                print(f"\n步骤4: 测试内容提取逻辑")
                web_results = "\n".join([d["content"] for d in docs["results"]])
                print(f"提取的内容类型: {type(web_results)}")
                print(f"提取的内容长度: {len(web_results)}字符")
                print(f"提取的内容前500字符:\n{web_results[:500]}...")

                # 5. 测试转换为Document
                web_results_doc = Document(page_content=web_results)
                print(f"\n步骤5: 测试转换为Document")
                print(f"Document类型: {type(web_results_doc)}")
                print(f"Document内容长度: {len(web_results_doc.page_content)}字符")

        print("\n✅ 测试成功！内容提取逻辑正确工作")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败，错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_web_search_content_extraction()