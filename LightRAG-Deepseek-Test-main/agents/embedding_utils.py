from sentence_transformers import SentenceTransformer
import torch

class EmbeddingUtil:
    def __init__(self, model_name="BAAI/bge-large-zh-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    async def get_embeddings(self, texts):
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings
    
    async def get_query_embedding(self, query):
        return await self.get_embeddings([query])[0]