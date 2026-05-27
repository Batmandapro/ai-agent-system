import json
import os
import numpy as np
import requests

# ── CONFIG ────────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
DB_PATH     = "data/cases_db.json"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def embed(text):
    """Embed a single text string using nomic-embed-text via Ollama."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
    except Exception:
        return None

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def _normalise_source(source: str) -> str:
    """
    Normalise a source string for case-name matching.
    Strips file extension, underscores, and common punctuation.
    Returns lowercase string suitable for substring matching.
    """
    s = source.lower()
    for ext in (".pdf", ".txt", ".docx"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    s = s.replace("_", " ").replace("-", " ")
    return s.strip()

# ── MAIN CLASS ────────────────────────────────────────────────────────────────

class LegalFAISS:
    def __init__(self, path=DB_PATH):
        self.path   = path
        self.chunks = self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("chunks", [])
            return []
        except Exception:
            return []

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def add(self, text, source_type, meta):
        """Embed and store a chunk."""
        vector = embed(text)
        if vector is None:
            return False
        self.chunks.append({
            "text":        text,
            "source_type": source_type,
            "meta":        meta,
            "vector":      vector
        })
        self._save()
        return True

    def search(self, query, top_k=6):
        """
        Return top_k chunks most similar to query using cosine similarity,
        searching across ALL stored chunks.
        """
        q_vec = embed(query)
        if q_vec is None:
            return []

        results = []
        for c in self.chunks:
            vec = c.get("vector")
            if not vec:
                continue
            score = cosine_similarity(q_vec, vec)
            results.append({
                "text":        c.get("text", ""),
                "meta":        c.get("meta", {}),
                "source_type": c.get("source_type", ""),
                "source":      c.get("source", c.get("meta", {}).get("source", "")),
                "score":       score
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_by_source(self, query, case_name_hint: str, top_k=8):
        """
        Return top_k chunks filtered to only those whose source filename
        contains the case_name_hint (case-insensitive substring match).

        Used for case_summary mode to prevent retrieval from unrelated cases.
        Falls back to unrestricted search if no chunks match the hint.
        """
        q_vec = embed(query)
        if q_vec is None:
            return []

        hint_lower = case_name_hint.lower().strip()

        matching = []
        for c in self.chunks:
            raw_source = c.get("source", c.get("meta", {}).get("source", ""))
            normalised = _normalise_source(raw_source)
            if hint_lower in normalised:
                vec = c.get("vector")
                if not vec:
                    continue
                score = cosine_similarity(q_vec, vec)
                matching.append({
                    "text":        c.get("text", ""),
                    "meta":        c.get("meta", {}),
                    "source_type": c.get("source_type", ""),
                    "source":      raw_source,
                    "score":       score
                })

        if not matching:
            print(f"[FAISS] No source match for '{case_name_hint}' — "
                  f"falling back to unrestricted search. "
                  f"Check that the case file name contains this name.")
            return self.search(query, top_k=top_k)

        matching.sort(key=lambda x: x["score"], reverse=True)
        return matching[:top_k]

    def count(self):
        """Return total number of stored chunks."""
        return len(self.chunks)

    def list_sources(self):
        """Return a deduplicated list of all source filenames in the database."""
        seen = set()
        sources = []
        for c in self.chunks:
            src = c.get("source", c.get("meta", {}).get("source", ""))
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
        return sorted(sources)