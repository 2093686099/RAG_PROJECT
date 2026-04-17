"""Graph2 调试 UI — 本地聊天界面，直接驱动 graph2.graph 管道。"""
import os

# 系统代理（如 Clash at 127.0.0.1:7890）会劫持 Gradio 的 localhost 自检，必须显式排除
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import gradio as gr

from graph2.graph_2 import graph
from utils.log_utils import log


def chat(message: str, history):
    if not message or not message.strip():
        yield "请输入问题"
        return

    trace = []
    final = ""
    try:
        for output in graph.stream({"question": message}):
            for node, value in output.items():
                trace.append(node)
                if isinstance(value, dict) and value.get("generation"):
                    final = value["generation"]
                yield f"⏳ 路径: {' → '.join(trace)}"

        body = final or "(流程结束但没有 generation)"
        yield f"{body}\n\n---\n🔍 调试路径: {' → '.join(trace)}"
    except Exception as e:
        log.exception(e)
        yield f"❌ {type(e).__name__}: {e}"


if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat,
        title="RAG Graph2 调试",
        description="Adaptive RAG · glm-5 · Qwen3-Embedding-8B · Milvus",
    )
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
