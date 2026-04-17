from utils.log_utils import log


def draw_graph(graph, file_name: str):
    """
    绘制LangGraph图并保存为PNG文件。

    Args:
        graph: LangGraph图对象
        filename (str): 输出PNG文件的路径
    """
    try:
        mermaid_code = graph.get_graph().draw_mermaid_png()
        with open(file_name, "wb") as f:
            f.write(mermaid_code)

    except Exception as e:
        log.exception(e)
