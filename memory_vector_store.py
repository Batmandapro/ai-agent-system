import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorMemoryStore:
    def __init__(self, path="data/vector_memory.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def _load(self):
        with open(self.path, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, text, metadata=None):
        data = self._load()

        embedding = self.model.encode(text).tolist()

        data.append({
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {}
        })

        self._save(data)

    def search(self, query, top_k=5):
        data = self._load()

        if not data:
            return []

        query_emb = self.model.encode(query)

        def cosine(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        scored = []
        for item in data:
            score = cosine(query_emb, item["embedding"])
            scored.append((score, item))

        scored.sort(reverse=True, key=lambda x: x[0])

        return [item for _, item in scored[:top_k]]