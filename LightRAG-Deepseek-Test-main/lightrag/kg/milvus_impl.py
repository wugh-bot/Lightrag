from ..base import BaseVectorStorage
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
from typing import Dict, Union

class MilvusVectorDBStorage(BaseVectorStorage):
    def __init__(self, namespace: str, global_config: dict, embedding_func=None, **kwargs):
        super().__init__(namespace, global_config, embedding_func)
        self.collection = self._init_collection()
        
    def _init_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
        ]
        schema = CollectionSchema(fields=fields, description=f"collection_{self.namespace}")
        
        if self.namespace not in utility.list_collections():
            collection = Collection(name=self.namespace, schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
            )
        else:
            collection = Collection(name=self.namespace)
            collection.load()
            
        return collection
        
    async def upsert(self, documents: dict):
        try:
            entities = []
            for doc_id, doc in documents.items():
                if "embedding" not in doc:
                    embeddings = await self.embedding_func([doc["content"]])
                    doc["embedding"] = embeddings[0]
                
                entities.append({
                    "doc_id": doc_id,
                    "content": doc["content"],
                    "embedding": doc["embedding"].tolist()
                })
            
            self.collection.insert(entities)
            return True
        except Exception as e:
            print(f"Milvus upsert error: {str(e)}")
            return False
            
    async def query(self, query: str, top_k=5):
        try:
            query_embedding = await self.embedding_func([query])
            self.collection.load()
            results = self.collection.search(
                data=[query_embedding[0].tolist()],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["doc_id", "content"]
            )
            
            return [
                {
                    "id": hit.doc_id,
                    "content": hit.content,
                    "distance": hit.distance
                }
                for hit in results[0]
            ]
        except Exception as e:
            print(f"Milvus query error: {str(e)}")
            return []

# 添加别名以兼容错误拼写
MilvusVectorDBStorge = MilvusVectorDBStorage
