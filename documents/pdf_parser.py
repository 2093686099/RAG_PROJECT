import base64
import json
import os
import subprocess
from io import BytesIO
from typing import List

import pypdfium2 as pdfium
from langchain_core.documents import Document

from documents.markdown_parser import MarkdownParser
from utils.log_utils import log


class PDFParser:
    """
    使用OCR视觉模型解析PDF文件：
    PDF -> 逐页渲染图片 -> glm-ocr识别为Markdown -> 复用MarkdownParser切割管道
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    OCR_MODEL = "glm-ocr:bf16"
    RENDER_SCALE = 200 / 72  # 200DPI，A4约3.87M像素，远低于模型9.6M上限
    OCR_TIMEOUT = 120  # 单页OCR超时秒数

    def __init__(self):
        self.markdown_parser = MarkdownParser()

    def _render_pdf_to_images(self, pdf_file: str) -> list:
        """将PDF每页渲染为PIL Image列表"""
        pdf = pdfium.PdfDocument(pdf_file)
        images = []
        for i in range(len(pdf)):
            try:
                page = pdf[i]
                bitmap = page.render(scale=self.RENDER_SCALE)
                pil_image = bitmap.to_pil()
                images.append((i + 1, pil_image))
            except Exception as e:
                log.error(f"PDF第{i + 1}页渲染失败: {e}")
        pdf.close()
        return images

    def _ocr_image_to_markdown(self, image, page_num: int) -> str:
        """调用glm-ocr视觉模型，将页面图片识别为Markdown文本（通过curl绕过Python HTTP库兼容性问题）"""
        # PIL Image -> PNG bytes -> base64
        buf = BytesIO()
        image.save(buf, format="PNG")
        base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = json.dumps({
            "model": self.OCR_MODEL,
            "prompt": "Text Recognition:\n请对这张图片进行OCR识别，输出结构化的Markdown格式文本。保留标题层级、段落结构和表格。",
            "images": [base64_str],
            "stream": False,
            "options": {"num_ctx": 16384}
        })

        try:
            result = subprocess.run(
                ["curl", "-s", "--noproxy", "localhost", "--max-time", str(self.OCR_TIMEOUT),
                 self.OLLAMA_URL, "-d", "@-", "-H", "Content-Type: application/json"],
                input=payload, capture_output=True, text=True
            )
            if result.returncode != 0:
                log.error(f"第{page_num}页OCR curl失败: {result.stderr}")
                return ""
            data = json.loads(result.stdout)
            if "error" in data:
                log.error(f"第{page_num}页OCR模型错误: {data['error']}")
                return ""
            content = data["response"]
            log.info(f"第{page_num}页OCR完成，输出{len(content)}字符")
            return content
        except Exception as e:
            log.error(f"第{page_num}页OCR失败: {e}")
            return ""

    def _get_md_path(self, pdf_file: str) -> str:
        """根据PDF路径生成对应的md缓存路径（同目录，同名.md）"""
        return os.path.splitext(pdf_file)[0] + '.md'

    def parser_pdf_to_documents(self, pdf_file: str, encoding='utf-8') -> List[Document]:
        """
        解析PDF文件为Document列表（公开方法，签名与旧版一致）

        流程：
        1. 逐页OCR，每完成一页立刻追加写入同名.md文件
        2. 全部OCR完成后，统一用MarkdownParser切割
        3. 如果.md文件已存在且非空，跳过OCR直接用缓存
        """
        pdf_abs = os.path.abspath(pdf_file)
        pdf_basename = os.path.basename(pdf_file)
        md_path = self._get_md_path(pdf_abs)

        # 如果md缓存已存在且非空，直接跳过OCR
        if os.path.exists(md_path) and os.path.getsize(md_path) > 0:
            log.info(f"发现OCR缓存，跳过OCR: {md_path}")
        else:
            log.info(f"开始OCR解析PDF: {pdf_file}")

            # 1. 渲染所有页面为图片
            page_images = self._render_pdf_to_images(pdf_file)
            log.info(f"PDF共{len(page_images)}页渲染完成")

            if not page_images:
                log.warning(f"PDF渲染结果为空: {pdf_file}")
                return []

            # 2. 逐页OCR，每完成一页立刻追加写入md文件
            with open(md_path, 'w', encoding='utf-8') as f:
                for page_num, image in page_images:
                    md_text = self._ocr_image_to_markdown(image, page_num)
                    if md_text.strip():
                        if f.tell() > 0:
                            f.write("\n\n")
                        f.write(md_text)
                        f.flush()

            if os.path.getsize(md_path) == 0:
                log.warning(f"所有页面OCR结果为空: {pdf_file}")
                os.unlink(md_path)
                return []

            log.info(f"OCR完成，已写入: {md_path}")

        # 3. 统一用MarkdownParser切割
        documents = self.markdown_parser.parse_markdown_to_documents(md_path)

        # 4. 修正metadata，指回原始PDF
        for doc in documents:
            doc.metadata['source'] = pdf_abs
            doc.metadata['filename'] = pdf_basename
            doc.metadata['filetype'] = 'application/pdf'

        log.info(f"PDF解析完成: {pdf_basename}，共{len(documents)}个文档块")
        return documents


if __name__ == '__main__':
    parser = PDFParser()
    # 测试用法：python documents/pdf_parser.py <pdf文件路径>
    import sys
    if len(sys.argv) > 1:
        docs = parser.parser_pdf_to_documents(sys.argv[1])
        for item in docs:
            print(f"元数据：{item.metadata}")
            print(f"内容：{item.page_content[:200]}\n")
            print("------" * 10)
    else:
        print("用法: python documents/pdf_parser.py <pdf文件路径>")
