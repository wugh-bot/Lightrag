import asyncio
import inspect
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, siliconcloud_embedding
from lightrag.lightrag import always_get_an_event_loop
from lightrag.utils import EmbeddingFunc
import numpy as np
import os
import json
# 在文件开头添加新的导入
import textract
from PyPDF2 import PdfReader
from docx import Document
from typing import AsyncGenerator

# 文件存储目录
FILE_STORAGE_DIR = "./file_storage"
if not os.path.exists(FILE_STORAGE_DIR):
    os.mkdir(FILE_STORAGE_DIR)
WORKING_DIR = "./working"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

class LightRAGAgent:
    def __init__(self):
        # 使用字符串形式获取环境变量
        self.llm_api_key = os.getenv("DEEPSEEK_API_KEY") or ""
        self.llm_model = os.getenv("DEEPSEEK_MODEL") or ""
        self.llm_base_url = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        self.embedding_model = os.getenv("SILICONCLOUD_EMBEDDING_MODEL") or "BAAI/bge-m3"
        self.embedding_api_key = os.getenv("SILICONCLOUD_API_KEY") or ""
        self.agent = None

    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            self.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            **kwargs
        )

    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        try:
            embeddings = await siliconcloud_embedding(
                texts,
                model=self.embedding_model,
                api_key=self.embedding_api_key,
            )
            if embeddings is None:
                raise ValueError("Embedding generation failed")
            return embeddings
        except Exception as e:
            print(f"Embedding 调用失败: {str(e)}")
            raise

    # 添加 get_embeddings 方法
    async def get_embeddings(self, file_paths: list[str]) -> np.ndarray:
        """获取文件的嵌入向量"""
        try:
            texts = []
            for file_path in file_paths:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    texts.append(text)
            
            # 使用 embedding_func 获取嵌入向量
            embeddings = await self.embedding_func(texts)
            return embeddings
        except Exception as e:
            print(f"获取文件嵌入向量失败: {str(e)}")
            raise
    
    # 添加 get_query_embedding 方法
    async def get_query_embedding(self, query: str) -> np.ndarray:
        """获取查询的嵌入向量"""
        try:
            embedding = await self.embedding_func([query])
            return embedding[0]  # 返回第一个向量
        except Exception as e:
            print(f"获取查询嵌入向量失败: {str(e)}")
            raise

    async def init_rag(self):
        """初始化 RAG 系统"""
        try:
            # 使用一个测试文本获取embedding维度
            test_text = ["This is a test sentence."]
            test_embedding = await self.embedding_func(test_text)
            embedding_dimension = test_embedding.shape[1]
            print(f"Detected embedding dimension: {embedding_dimension}")
    
            WORKING_DIR = "./working"  # 确保这个目录存在
            if not os.path.exists(WORKING_DIR):
                os.makedirs(WORKING_DIR)
    
            # 使用 NanoVectorDBStorage 替代 MilvusVectorDBStorage
            self.agent = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=self.llm_model_func,
                llm_model_max_token_size=32768,
                llm_model_max_async=64,
                addon_params={"language": "Simplified Chinese"},
                chunk_token_size=1024,
                chunk_overlap_token_size=100,
                embedding_func=EmbeddingFunc(
                    embedding_dim=embedding_dimension,
                    max_token_size=8192,
                    func=self.embedding_func,
                ),
                embedding_batch_num=10,
                vector_storage="NanoVectorDBStorage",  # 使用已知存在的向量存储实现
            )
            self.agent.__post_init__()
            return True
        except Exception as e:
            print(f"初始化 RAG 系统失败: {str(e)}")
            return False

    def get_doc_id(self):
        try:
            with open(f"{WORKING_DIR}/kv_store_doc_status.json", 'r', encoding='utf-8') as file:
                data = json.load(file)
            return list(data.keys())
        except Exception as e:
            print(f"lightrag get doc id error occurred: {e}")
            return None

    async def insert_file(self, file_paths):
        contents = []
        for file_path in file_paths:
            try:
                if file_path.lower().endswith('.txt'):
                    # 处理 txt 文件
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()
                elif file_path.lower().endswith('.pdf'):
                    # 处理 PDF 文件
                    text_content = ""
                    with open(file_path, 'rb') as file:
                        pdf_reader = PdfReader(file)
                        for page in pdf_reader.pages:
                            text_content += page.extract_text() + "\n"
                elif file_path.lower().endswith(('.doc', '.docx')):
                    # 处理 Word 文档
                    doc = Document(file_path)
                    text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                else:
                    # 其他类型文件使用 textract 处理
                    text_content = textract.process(file_path).decode('utf-8', errors='ignore')
                
                if not text_content.strip():
                    print(f"Warning: No text content extracted from {file_path}")
                    return False
                    
                contents.append(text_content)
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                return False
                
        try:
            await self.agent.ainsert(contents)
            return True
        except Exception as e:
            print(f"lightrag insert error occurred: {e}")
            return False

    async def get_summarize(self, input_text) -> str:
        response = await self.agent.aquery(
                input_text, param=QueryParam(mode="hybrid", top_k=60)
            )
        # print(f"lightrag response: {response}")
        return response

    async def delete_file(self, file_name):
        try:
            await self.agent.adelete_by_doc_id(file_name)
            return True
        except Exception as e:
            print(f"lightrag delete error occurred: {e}")
            return False