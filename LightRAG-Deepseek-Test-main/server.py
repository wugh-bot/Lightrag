# 在导入部分添加CORS支持，解决可能的跨域问题
from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware  # 添加CORS中间件
from typing import AsyncGenerator
from pydantic import BaseModel
import uvicorn
import os
import sys
from typing import Optional
import json
import asyncio
from dotenv import load_dotenv
# 添加 Milvus 相关导入
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import faiss
import pickle  # 添加pickle用于保存FAISS索引

# 添加 Milvus 配置常量
MILVUS_HOST = "localhost"  # Milvus 服务器地址
MILVUS_PORT = 19530       # Milvus 服务端口
COLLECTION_NAME = "document_store"  # collection 名称
VECTOR_DIM = 1024         # 向量维度
FAISS_INDEX_PATH = "./faiss_index.bin"  # FAISS索引保存路径

load_dotenv()  # 加载 .env 文件中的环境变量

from agents.lightrag import LightRAGAgent

FILE_STORAGE_DIR = "./file_storage"
# 在应用启动时创建文件存储目录
if not os.path.exists(FILE_STORAGE_DIR):
    os.makedirs(FILE_STORAGE_DIR)

app = FastAPI()

# 修改CORS中间件配置，确保正确处理OPTIONS请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 明确列出所有允许的方法
    allow_headers=["*"],  # 允许所有头
    expose_headers=["*"],  # 暴露所有头部
)

# 定义QueryRequest类（只保留一个定义）
class QueryRequest(BaseModel):
    query: str

# 全局变量定义
rag = None

# 根路径返回HTML页面
@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>LightRAG API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    line-height: 1.6;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                h1 {
                    color: #333;
                }
                a {
                    display: inline-block;
                    margin: 10px 0;
                    padding: 10px 15px;
                    background-color: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                }
                a:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>欢迎使用 LightRAG API</h1>
                <p>这是一个基于LightRAG的文档问答系统。</p>
                <a href="/docs">查看API文档</a>
                <a href="/redoc">查看ReDoc文档</a>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 重定向/doc到/docs
@app.get("/doc")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# 添加OPTIONS请求处理
@app.options("/{path:path}")
async def options_handler(path: str):
    return {"status": "ok"}

# 应用启动事件
@app.on_event("startup")
async def startup_event():
    global rag
    
    # 初始化RAG系统
    for attempt in range(3):  # 尝试3次初始化
        try:
            rag = LightRAGAgent()
            init_success = await rag.init_rag()
            if init_success:
                print(f"RAG系统初始化成功 (尝试 {attempt+1}/3)")
                break
            else:
                print(f"RAG系统初始化失败 (尝试 {attempt+1}/3)，将重试...")
                await asyncio.sleep(2)  # 等待2秒后重试
        except Exception as e:
            print(f"RAG系统初始化出错 (尝试 {attempt+1}/3): {str(e)}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(2)  # 等待2秒后重试
    
    if rag is None:
        print("所有初始化尝试均失败，服务可能无法正常工作")
        return
    
    # 自动加载已有文件
    try:
        files = os.listdir(FILE_STORAGE_DIR)
        if files:
            print(f"发现 {len(files)} 个已有文件，正在加载...")
            # 添加超时处理，避免程序卡住
            try:
                # 只处理文本文件，避免处理复杂文件导致卡住
                text_files = [f for f in files if f.lower().endswith('.txt')]
                if text_files:
                    print(f"将优先处理 {len(text_files)} 个文本文件...")
                    file_paths = [os.path.join(FILE_STORAGE_DIR, f) for f in text_files]
                    # 设置超时时间为60秒
                    load_result = await asyncio.wait_for(rag.insert_file(file_paths), timeout=60)
                    if load_result:
                        print("文本文件加载成功")
                    else:
                        print("警告：文件加载可能不完整")
                else:
                    print("没有找到文本文件，跳过自动加载")
            except asyncio.TimeoutError:
                print("文件加载超时，将在请求时按需加载文件")
            except Exception as e:
                print(f"文件加载过程中出错: {str(e)}")
                print("将在请求时按需加载文件")
            
            # 验证文件是否已正确加载
            try:
                doc_ids = rag.get_doc_id()
                print(f"RAG系统中的文档ID: {doc_ids}")
                print(f"文档ID数量: {len(doc_ids)}, 文件数量: {len(files)}")
            except Exception as e:
                print(f"获取文档ID时出错: {str(e)}")
    except Exception as e:
        print(f"加载文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()

    # 初始化向量存储
    try:
        # Milvus 连接
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        
        # 检查集合是否存在，如果存在则使用，不存在则创建
        if COLLECTION_NAME not in utility.list_collections():
            # 创建 collection schema - 启用动态字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=200),  # 预定义字段
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=200),  # 预定义字段
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),  # 预定义字段
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
            ]
            schema = CollectionSchema(
                fields=fields, 
                description="document store"
            )
            
            # 创建集合
            collection = Collection(name=COLLECTION_NAME, schema=schema)
            
            # 创建索引
            collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
            )
        else:
            # 使用现有集合
            collection = Collection(name=COLLECTION_NAME)
            # 确保索引已加载
            collection.load()
        
        # 初始化或加载FAISS索引
        if os.path.exists(FAISS_INDEX_PATH):
            print(f"加载FAISS索引: {FAISS_INDEX_PATH}")
            with open(FAISS_INDEX_PATH, 'rb') as f:
                faiss_index = pickle.load(f)
            print(f"FAISS索引加载成功，包含 {faiss_index.ntotal} 个向量")
        else:
            print("创建新的FAISS索引")
            faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
        
        # 打印集合信息
        print(f"Milvus 集合 {COLLECTION_NAME} 已连接，当前实体数量: {collection.num_entities}")
        
        # 如果Milvus中有数据但FAISS为空，则从Milvus加载向量到FAISS
        if collection.num_entities > 0 and faiss_index.ntotal == 0:
            print("从Milvus加载向量到FAISS...")
            # 查询所有向量
            results = collection.query(
                expr="id >= 0",  # 查询所有记录
                output_fields=["embedding"],
                consistency_level="Strong"
            )
            
            if results:
                # 提取向量并添加到FAISS
                vectors = np.array([r['embedding'] for r in results])
                faiss_index.add(vectors)
                print(f"成功从Milvus加载 {len(vectors)} 个向量到FAISS")
                
                # 保存FAISS索引
                with open(FAISS_INDEX_PATH, 'wb') as f:
                    pickle.dump(faiss_index, f)
                print(f"FAISS索引已保存到: {FAISS_INDEX_PATH}")
        
    except Exception as e:
        print(f"向量数据库初始化错误：{str(e)}")

# 列出文件列表
@app.get("/file/list/")
async def list_files():
    global rag
    if rag is None:
        return {"error": "RAG系统未初始化"}
        
    files = os.listdir(FILE_STORAGE_DIR)
    if len(files) == 0:
        return {"files": []}
    else:
        files_id = rag.get_doc_id()
        
        return {"files": 
            [{"id": file_id, "name": file_name} for file_id, file_name in zip(files_id, files)]
        }

# 修改文件处理函数，增强错误处理和文件类型支持
# 修复 process_file 函数，删除重复定义并修正语法错误
# 修改文件处理函数中的摘要生成部分
async def process_file(files: list[UploadFile], text: str):
    global rag
    if rag is None:
        return [{"summary": "RAG系统未初始化"}]
        
    try:
        results = []
        file_paths = []
        processed_files = []

        # 检查文件是否为空
        if not files:
            return [{"summary": "没有提供文件"}]

        # 先保存所有文件到本地
        for file in files:
            try:
                # 确保文件指针在开始位置
                await file.seek(0)
                content = await file.read()
                if not content:  # 检查文件内容是否为空
                    print(f"警告: 文件 {file.filename} 内容为空")
                    continue
                
                # 获取文件扩展名并转换为小写
                ext = os.path.splitext(file.filename)[1].lower()
                
                # 处理文件名中的特殊字符，避免文件系统问题
                safe_filename = "".join([c for c in file.filename if c.isalnum() or c in "._- "]).rstrip()
                if not safe_filename:
                    safe_filename = f"file_{len(file_paths)}{ext}"
                
                file_path = os.path.join(FILE_STORAGE_DIR, safe_filename)
                
                # 如果文件已存在，添加时间戳避免覆盖
                if os.path.exists(file_path):
                    import time
                    timestamp = int(time.time())
                    name_part, ext_part = os.path.splitext(safe_filename)
                    safe_filename = f"{name_part}_{timestamp}{ext_part}"
                    file_path = os.path.join(FILE_STORAGE_DIR, safe_filename)
                
                with open(file_path, "wb") as f:
                    f.write(content)
                
                file_paths.append(file_path)
                processed_files.append({"original_name": file.filename, "saved_as": safe_filename})
                print(f"成功保存文件: {file_path}")
            except Exception as e:
                print(f"保存文件 {file.filename} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if not file_paths:  # 如果没有成功保存任何文件
            return [{"summary": "所有文件处理失败"}]
        
        # 优化：根据文件数量决定处理策略
        successful_files = []
        failed_files = []
        
        # 如果文件数量少于3个，可以一次性处理，提高速度
        if len(file_paths) <= 3:
            try:
                print(f"批量处理 {len(file_paths)} 个文件")
                # 设置超时时间为180秒
                level = await asyncio.wait_for(rag.insert_file(file_paths), timeout=180)
                
                if level == True:
                    successful_files = processed_files
                    print(f"所有文件处理成功")
                else:
                    # 如果批量处理失败，回退到单文件处理
                    print("批量处理失败，切换到单文件处理模式")
                    for i, file_path in enumerate(file_paths):
                        try:
                            single_level = await asyncio.wait_for(rag.insert_file([file_path]), timeout=120)
                            if single_level == True:
                                successful_files.append(processed_files[i])
                            else:
                                failed_files.append(processed_files[i])
                            # 短暂延迟，但比原来更短
                            await asyncio.sleep(0.5)
                        except Exception as e:
                            failed_files.append(processed_files[i])
                            print(f"处理文件 {file_path} 时出错: {str(e)}")
            except Exception as e:
                print(f"批量处理文件时出错: {str(e)}")
                # 回退到单文件处理
                for i, file_path in enumerate(file_paths):
                    try:
                        single_level = await asyncio.wait_for(rag.insert_file([file_path]), timeout=120)
                        if single_level == True:
                            successful_files.append(processed_files[i])
                        else:
                            failed_files.append(processed_files[i])
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        failed_files.append(processed_files[i])
                        print(f"处理文件 {file_path} 时出错: {str(e)}")
        else:
            # 文件数量较多时，使用单文件处理，确保稳定性
            for i, file_path in enumerate(file_paths):
                try:
                    print(f"处理文件 {i+1}/{len(file_paths)}: {file_path}")
                    level = await asyncio.wait_for(rag.insert_file([file_path]), timeout=120)
                    
                    if level == True:
                        successful_files.append(processed_files[i])
                        print(f"文件 {file_path} 处理成功")
                    else:
                        failed_files.append(processed_files[i])
                        print(f"文件 {file_path} 处理失败")
                    
                    # 减少延迟时间，提高响应速度
                    await asyncio.sleep(1)
                except Exception as e:
                    failed_files.append(processed_files[i])
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
        
        # 只在处理完所有文件后执行一次垃圾回收，而不是每个文件后都执行
        import gc
        gc.collect()
        
        # 如果有成功处理的文件，尝试生成摘要
        if successful_files:
            try:
                print(f"开始生成摘要，提示文本: {text}")
                # 增加摘要生成的超时时间到120秒
                try:
                    summary = await asyncio.wait_for(rag.get_summarize(text), timeout=120)
                    print(f"摘要生成成功: {summary[:100]}...")
                    
                    results.append({
                        "successful_files": successful_files,
                        "failed_files": failed_files,
                        "summary": summary
                    })
                except asyncio.TimeoutError:
                    print("摘要生成超时")
                    results.append({
                        "successful_files": successful_files,
                        "failed_files": failed_files,
                        "summary": "摘要生成超时，请尝试简化提示或减少文件数量。"
                    })
            except Exception as e:
                print(f"生成摘要时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # 尝试使用备用方法生成简单摘要
                try:
                    print("尝试使用备用方法生成简单摘要...")
                    # 如果get_summarize失败，尝试使用get_answer作为备用
                    backup_summary = await asyncio.wait_for(
                        rag.get_answer("请简要总结这些文档的主要内容"), 
                        timeout=120
                    )
                    results.append({
                        "successful_files": successful_files,
                        "failed_files": failed_files,
                        "summary": f"[备用摘要] {backup_summary}"
                    })
                    print(f"备用摘要生成成功: {backup_summary[:100]}...")
                except Exception as backup_error:
                    print(f"备用摘要生成也失败: {str(backup_error)}")
                    results.append({
                        "successful_files": successful_files,
                        "failed_files": failed_files,
                        "summary": f"文件处理成功，但生成摘要时出错: {str(e)}"
                    })
        else:
            results.append({
                "successful_files": [],
                "failed_files": failed_files,
                "summary": "所有文件处理失败，无法生成摘要。"
            })

        return results
    except Exception as e:
        print(f"文件处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return [{"summary": f"处理文件时出错：{str(e)}"}]

# 修改文件上传接口，增加更多的错误处理和调试信息
# 修改文件上传接口
@app.post("/file/add_and_summarize/")
async def add_file_and_summarize(files: list[UploadFile] = File(None), text: str = Form(None)):
    global rag
    if rag is None:
        # 尝试重新初始化RAG系统
        try:
            rag = LightRAGAgent()
            init_success = await rag.init_rag()
            if not init_success:
                raise Exception("RAG系统初始化失败")
            print("已成功重新初始化RAG系统")
        except Exception as e:
            print(f"重新初始化RAG系统失败: {str(e)}")
            raise HTTPException(status_code=500, detail="RAG系统未初始化，请稍后再试")
    
    try:
        # 检查是否有文件上传
        if not files or len(files) == 0 or all(file.filename == "" for file in files):
            # 如果没有文件上传，返回已加载的文件信息
            loaded_files = os.listdir(FILE_STORAGE_DIR)
            try:
                doc_ids = rag.get_doc_id()
                return {
                    "message": "没有上传新文件，返回已加载的文件信息",
                    "loaded_files": loaded_files,
                    "doc_ids": doc_ids
                }
            except Exception as e:
                print(f"获取文档ID时出错: {str(e)}")
                return {
                    "message": "没有上传新文件，返回已加载的文件信息",
                    "loaded_files": loaded_files,
                    "error": f"获取文档ID时出错: {str(e)}"
                }
        
        # 打印上传的文件信息
        print(f"收到 {len(files)} 个文件:")
        for file in files:
            print(f"  - 文件名: {file.filename}, 内容类型: {file.content_type}")
        
        # 检查文件类型
        allowed_types = [".txt", ".pdf", ".doc", ".docx"]
        valid_files = []
        invalid_files = []
        
        for file in files:
            if not file.filename:  # 检查文件名是否为空
                print(f"警告: 跳过空文件名")
                continue
            
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in allowed_types:
                print(f"警告: 不支持的文件类型: {ext}, 文件: {file.filename}")
                invalid_files.append({"filename": file.filename, "reason": f"不支持的文件类型: {ext}"})
                continue
            
            # 检查文件大小，限制为10MB
            try:
                file_size = 0
                file.file.seek(0, 2)  # 移动到文件末尾
                file_size = file.file.tell()  # 获取文件大小
                file.file.seek(0)  # 重置文件指针到开头
                
                if file_size > 10 * 1024 * 1024:  # 10MB
                    print(f"警告: 文件过大: {file.filename}, 大小: {file_size/1024/1024:.2f}MB")
                    invalid_files.append({"filename": file.filename, "reason": "文件过大，超过10MB限制"})
                    continue
            except Exception as e:
                print(f"检查文件大小时出错: {str(e)}")
            
            valid_files.append(file)
        
        if not valid_files:
            return {
                "message": "没有有效的文件上传",
                "invalid_files": invalid_files,
                "error": "请检查文件类型和大小"
            }
        
        # 如果text参数为空，设置一个默认值
        if not text:
            text = "请总结这些文档的内容"
        
        print(f"开始处理 {len(valid_files)} 个有效文件，文本提示：{text}")
        
        # 处理文件 - 一次只处理一个文件，避免批处理问题
        results = await process_file(valid_files, text)
        
        # 添加无效文件信息到结果中
        if invalid_files and results:
            results[0]["invalid_files"] = invalid_files
        
        # 检查摘要是否生成成功
        if results and "summary" in results[0]:
            summary = results[0]["summary"]
            if summary.startswith("文件处理成功，但生成摘要时出错"):
                # 记录摘要生成失败
                print("警告: 摘要生成失败，但文件处理成功")
                # 添加更多的诊断信息
                results[0]["summary_status"] = "failed"
                # 尝试获取系统状态信息
                try:
                    import psutil
                    results[0]["system_info"] = {
                        "memory_percent": psutil.virtual_memory().percent,
                        "cpu_percent": psutil.cpu_percent(interval=0.1)
                    }
                except ImportError:
                    print("psutil模块未安装，无法获取系统信息")
                except Exception as e:
                    print(f"获取系统信息时出错: {str(e)}")
            else:
                results[0]["summary_status"] = "success"
        
        return results
    except HTTPException as e:
        # 直接重新抛出HTTP异常
        raise e
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")
        # 记录详细的错误信息
        import traceback
        traceback.print_exc()
        # 添加返回值，避免未定义返回
        raise HTTPException(status_code=500, detail=f"处理文件时发生错误: {str(e)}")

# 修改问答接口，增强错误处理和资源释放
# 修改问答接口中的方法调用
# 在导入部分添加
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.post("/qa/")
async def question_answering(request: QueryRequest):
    global rag
    if rag is None:
        # 使用 HTTPException 返回更规范的错误
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        print(f"收到问题: {request.query}")

        # 检查是否有已加载的文件
        files = os.listdir(FILE_STORAGE_DIR)
        if not files:
            # 如果没有文件，也应该返回错误，或者根据需求处理
            raise HTTPException(status_code=400, detail="No files in the system. Please upload files first.")

        # 检测查询语言
        is_english_query = is_english(request.query)

        # 使用get_summarize方法，但根据查询语言提供不同的提示
        try:
            print("开始生成回答...")

            # 根据查询语言构建提示
            if is_english_query:
                # 尝试简化英文提示
                legal_query = f"Based on the documents, answer the question: {request.query}"
                # 或者甚至更简单，如果 rag.get_summarize 能直接处理问题的话
                # legal_query = request.query
            else:
                legal_query = f"""
                请根据已加载的文档，准确回答以下问题。
                只提供与问题直接相关的内容，不要添加任何额外信息。
                如果文档中没有相关信息，请明确说明。

                问题: {request.query}
                """

            answer = await asyncio.wait_for(rag.get_summarize(legal_query), timeout=180)

            if not answer or answer.strip() == "":
                answer = "No relevant information found." if is_english_query else "无法找到相关信息，请尝试重新提问或提供更多细节。"

            print(f"回答生成成功: {answer[:100]}...")

            return {
                "question": request.query,
                "answer": answer
            }
        except asyncio.TimeoutError:
            # 超时也使用 HTTPException
            error_message = "Response generation timed out." if is_english_query else "回答生成超时，请尝试简化问题。"
            raise HTTPException(status_code=504, detail=error_message)
        except Exception as e:
            print(f"生成回答时出错: {str(e)}")
            # 生成回答时的内部错误
            error_message = f"Error generating response: {str(e)}" if is_english_query else f"生成回答时出错: {str(e)}"
            raise HTTPException(status_code=500, detail=error_message)
    except HTTPException as http_exc:
        # 直接重新抛出已知的 HTTP 异常
        raise http_exc
    except Exception as e:
        print(f"问答过程中出错: {str(e)}")
        # 其他未预料的内部错误
        error_message = f"Error processing question: {str(e)}" if is_english_query else f"处理问题时出错: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

# 添加语言检测函数
def is_english(text):
    """
    简单检测文本是否为英文
    """
    # 英文字符比例超过50%则认为是英文
    english_chars = sum(1 for c in text if ord('a') <= ord(c.lower()) <= ord('z'))
    return english_chars / max(len(text), 1) > 0.5

# 删除文件接口
@app.delete("/file/delete/")
async def delete_file(filename: str):
    global rag
    if rag is None:
        return {"error": "RAG系统未初始化"}
        
    try:
        files = os.listdir(FILE_STORAGE_DIR)
        if len(files) == 0:
            return {"error": "存储目录中没有找到文件。"}

        files_id = rag.get_doc_id()
        file_id_to_delete = None
        
        # 查找要删除的文件ID
        for file_id, file_name in zip(files_id, files):
            if file_name == filename:
                file_id_to_delete = file_id
                break
                
        if not file_id_to_delete:
            return {"error": f"文件 '{filename}' 未在RAG系统中找到。"}
            
        # 从RAG系统中删除文件
        tag = await rag.delete_file(file_id_to_delete)
        
        if tag == True:
            # 从文件系统中删除文件
            file_path = os.path.join(FILE_STORAGE_DIR, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                return {"message": f"文件 '{filename}' 删除成功。"}
            else:
                return {"error": f"文件 '{filename}' 在文件系统中不存在。"}
        else:
            return {"error": "文件删除失败，请检查服务器实现方式。"}
    except Exception as e:
        print(f"删除文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"删除文件时出错: {str(e)}")

# 健康检查接口
@app.get("/health")
async def health_check():
    global rag
    if rag is None:
        return {"status": "Error", "message": "RAG系统未初始化"}
        
    try:
        # 检查RAG系统
        response = await rag.test_funcs()
        
        # 检查文件系统
        files = os.listdir(FILE_STORAGE_DIR)
        
        # 检查文档ID
        files_id = rag.get_doc_id()
        
        # 返回健康状态
        if len(response) == 2:
            status = "OK"
        else:
            status = "LLM Error"
            
        if len(files) != len(files_id):
            status = "File Error"
            
        return {
            "status": status,
            "files_count": len(files),
            "doc_ids_count": len(files_id),
            "response": response
        }
    except Exception as e:
        print(f"健康检查出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "Error",
            "error": str(e)
        }

# 添加一个简单的测试接口
@app.get("/test")
async def test_api():
    return {"status": "API正常工作"}

# 修改主函数，增加工作线程数量
# 添加应用关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    global rag
    
    print("正在关闭服务器，释放资源...")
    
    # 关闭Milvus连接
    try:
        connections.disconnect("default")
        print("Milvus连接已关闭")
    except Exception as e:
        print(f"关闭Milvus连接时出错: {str(e)}")
    
    # 释放RAG资源
    try:
        if rag is not None:
            # 如果LightRAGAgent有关闭方法，调用它
            if hasattr(rag, 'close') and callable(getattr(rag, 'close')):
                await rag.close()
            # 将rag设置为None
            rag = None
            print("RAG系统资源已释放")
    except Exception as e:
        print(f"释放RAG系统资源时出错: {str(e)}")
    
    # 强制垃圾回收
    try:
        import gc
        gc.collect()
        print("垃圾回收已执行")
    except Exception as e:
        print(f"执行垃圾回收时出错: {str(e)}")

# 修改主函数，增加工作线程数量
if __name__ == "__main__":
    try:
        print("正在启动服务器，监听地址: 0.0.0.0:20000...")
        # 添加更多的启动参数，确保服务器能够处理并发请求
        uvicorn.run(
            app, 
            host="0.0.0.0",  # 绑定到所有网络接口
            port=20000,
            log_level="info",  # 增加日志级别
            reload=False,  # 禁用自动重载以避免潜在问题
            workers=1,  # 使用单进程模式，避免多进程导致的资源竞争
            timeout_keep_alive=120,  # 增加保持连接的超时时间
            limit_concurrency=10,  # 限制并发请求数量
            timeout_graceful_shutdown=30  # 优雅关闭的超时时间
        )
    except Exception as e:
        print(f"服务器启动失败: {str(e)}")
        import traceback
        traceback.print_exc()

# 添加摘要测试接口
@app.post("/test/summarize")
async def test_summarize(request: dict):
    global rag
    if rag is None:
        return {"error": "RAG系统未初始化"}
    
    try:
        text = request.get("text", "请总结已加载的文档内容")
        print(f"测试摘要生成，提示文本: {text}")
        
        # 检查是否有已加载的文件
        files = os.listdir(FILE_STORAGE_DIR)
        if not files:
            return {"error": "系统中没有任何文件，请先上传文件。"}
        
        # 检查RAG系统中的文档
        doc_ids = rag.get_doc_id()
        if not doc_ids or len(doc_ids) == 0:
            return {"error": "RAG系统中没有文档ID，请先上传并处理文件。"}
        
        # 测试摘要生成
        try:
            start_time = time.time()
            summary = await asyncio.wait_for(rag.get_summarize(text), timeout=120)
            end_time = time.time()
            
            return {
                "success": True,
                "summary": summary,
                "time_taken": f"{end_time - start_time:.2f}秒"
            }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "摘要生成超时",
                "suggestion": "请尝试简化提示或减少文件数量"
            }
        except Exception as e:
            print(f"摘要生成测试出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    except Exception as e:
        print(f"摘要测试接口出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/qa/raw/")
async def question_answering_raw(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "")
        if not query or query.strip() == "":
            return {"error": "请提供有效的问题"}
        
        query_request = QueryRequest(query=query)
        return await question_answering(query_request)
    except json.JSONDecodeError:
        return {"error": "无效的JSON格式"}
    except Exception as e:
        print(f"处理原始问答请求时出错: {str(e)}")
        return {"error": str(e)}

@app.get("/qa/get/")
async def question_answering_get(query: str = ""):
    if not query or query.strip() == "":
        return {"error": "请提供有效的问题"}
    
    try:
        query_request = QueryRequest(query=query)
        return await question_answering(query_request)
    except Exception as e:
        print(f"GET问答接口出错: {str(e)}")
        return {"error": str(e)}
