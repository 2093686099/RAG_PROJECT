import argparse
import multiprocessing
import os
from multiprocessing import Queue

from documents.milvus_db import MilvusVectorSave
from documents.pdf_parser import PDFParser
from utils.log_utils import log

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datas', 'pdf')


def file_parser_process(dir_path: str, output_queue: Queue,
                         skip_filenames: set, batch_size: int = 20):
    """解析进程：扫描目录下的 PDF 文件，跳过已入库的，分批放入队列"""
    log.info(f"解析进程开始扫描目录: {dir_path}")

    pdf_files = [
        os.path.join(dir_path, f)
        for f in sorted(os.listdir(dir_path))
        if f.endswith(".pdf")
    ]

    if not pdf_files:
        log.warning("未找到任何 .pdf 文件")
        output_queue.put(None)
        return

    # 增量模式：跳过已入库的文件
    if skip_filenames:
        before = len(pdf_files)
        pdf_files = [f for f in pdf_files if os.path.basename(f) not in skip_filenames]
        skipped = before - len(pdf_files)
        if skipped:
            log.info(f"跳过 {skipped} 个已入库的文件")

    if not pdf_files:
        log.info("没有需要处理的新文件")
        output_queue.put(None)
        return

    parser = PDFParser()
    doc_batch = []
    for file_path in pdf_files:
        try:
            log.info(f"正在解析: {os.path.basename(file_path)}")
            docs = parser.parser_pdf_to_documents(file_path)
            if docs:
                doc_batch.extend(docs)

            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch.copy())
                doc_batch.clear()
        except Exception as e:
            log.error(f"解析失败 {file_path}: {e}")
            log.exception(e)

    if doc_batch:
        output_queue.put(doc_batch.copy())

    output_queue.put(None)
    log.info(f"解析完成，共处理 {len(pdf_files)} 个文件")


def milvus_writer_process(input_queue: Queue):
    """写入进程：从队列读取文档批次并写入 Milvus"""
    log.info("Milvus 写入进程启动")

    mv = MilvusVectorSave()
    mv.create_connection()
    total_count = 0
    while True:
        try:
            datas = input_queue.get()
            if datas is None:
                break
            if isinstance(datas, list):
                mv.add_documents(datas)
                total_count += len(datas)
                log.info(f"累计已写入: {total_count} 个文档块")
        except Exception as e:
            log.error(f"写入 Milvus 失败: {e}")
            log.exception(e)

    log.info(f"写入进程结束，共写入 {total_count} 个文档块")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将 PDF 文件解析并写入 Milvus")
    parser.add_argument('--rebuild', action='store_true',
                        help='重建 collection（清空已有数据）')
    parser.add_argument('--dir', default=DATA_DIR,
                        help=f'PDF 文件目录（默认: {DATA_DIR}）')
    args = parser.parse_args()

    mv = MilvusVectorSave()
    if args.rebuild:
        log.info("--rebuild 模式：删除旧表并重建")
        mv.create_collection()
        skip_filenames = set()
    else:
        mv.ensure_collection()
        skip_filenames = mv.get_existing_filenames()
        if skip_filenames:
            log.info(f"增量模式：已入库 {len(skip_filenames)} 个文件")

    docs_queue = Queue(maxsize=20)

    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(args.dir, docs_queue, skip_filenames)
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,)
    )

    parser_proc.start()
    writer_proc.start()

    parser_proc.join()
    writer_proc.join()

    log.info("所有任务完成")
