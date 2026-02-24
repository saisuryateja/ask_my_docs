import faiss
import numpy as np
from pathlib import Path


class VectorStore:
    def __init__(self, embeddings: np.ndarray):
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings)

    def save(self, path: Path):
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        instance = cls.__new__(cls)
        instance.index = faiss.read_index(str(path))
        instance.dim = instance.index.d
        return instance

    def search(self, query_embedding: np.ndarray, top_k: int):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        return indices[0], distances[0]