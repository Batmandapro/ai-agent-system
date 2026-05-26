import json
import os
import numpy as np


class LegalFAISS:
    def __init__(self, path="data/cases_db.json"):
        self.path = path
        self.data = self._load()

        # Ensure schema consistency
        if not isinstance(self.data, dict):
            self.data = {"chunks": []}

        if "chunks" not in self.data:
            self.data["chunks"] = []

    # ----------------------------
    # LOAD / SAVE
    # ----------------------------

    def _load(self):
        if not os.path.exists(self.path):
            return {"chunks": []}

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Fix legacy bad format (list instead of dict)
            if isinstance(data, list):
                return {"chunks": data}

            if isinstance(data, dict):
                return data

            return {"chunks": []}

        except Exception:
            return {"chunks": []}

    def _save(self):
        tmp_path = self.path + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        os.replace(tmp_path, self.path)

    # ----------------------------
    # CORE API
    # ----------------------------

    def add(self, text, source_type, meta):
        chunk = {
            "text": text,
            "source_type": source_type,
            "meta": meta,
            "vector": self._embed(text)
        }

        self.data["chunks"].append(chunk)
        self._save()

    def search(self, query, top_k=5):
        q_vec = self._embed(query)

        results = []

        for c in self.data["chunks"]:
            score = self._score(q_vec, c.get("vector", []))

            results.append({
                "text": c.get("text", ""),
                "meta": c.get("meta", {}),
                "source_type": c.get("source_type", ""),
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ----------------------------
    # EMBEDDING (TEMP SIMPLE MODEL)
    # ----------------------------

    def _embed(self, text):
        """
        Lightweight placeholder embedding.
        (Replace later with Ollama or sentence-transformers)
        """
        return [hash(w) % 1000 for w in text.lower().split()[:64]]

    def _score(self, a, b):
        if not a or not b:
            return 0.0

        n = min(len(a), len(b))
        if n == 0:
            return 0.0

        return float(np.dot(a[:n], b[:n]))