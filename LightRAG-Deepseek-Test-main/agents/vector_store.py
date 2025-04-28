from pymilvus import Collection, utility
import faiss
import numpy as np
from typing import List, Dict, Any
import asyncio

class VectorStore:
    def __init__(self, collection: Collection, faiss_index: faiss.Index):
        self.collection = collection
        self.faiss_index = faiss_index
        
    async def insert_vectors(self, doc_id: str, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        try:
            # 存入 Milvus
            entities = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                entities.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "content": chunk["content"],
                    "embedding": embedding.tolist()
                })
            self.collection.insert(entities)
            
            # 更新 FAISS
            self.faiss_index.add(embeddings)
            return True
        except Exception as e:
            print(f"插入向量失败：{str(e)}")
            return False
            
    async def search(self, query_vector: np.ndarray, top_k: int = 5):
        try:
            # Milvus 搜索
            self.collection.load()
            milvus_results = self.collection.search(
                data=[query_vector.tolist()],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                output_fields=["doc_id", "chunk_id", "content"],
                limit=top_k
            )
            
            # FAISS 搜索
            faiss_distances, faiss_indices = self.faiss_index.search(
                np.array([query_vector]), top_k
            )
            
            return {
                "milvus_results": milvus_results[0],
                "faiss_results": {
                    "distances": faiss_distances[0].tolist(),
                    "indices": faiss_indices[0].tolist()
                }
            }
        except Exception as e:
            print(f"搜索向量失败：{str(e)}")
            return None