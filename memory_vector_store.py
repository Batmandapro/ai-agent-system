# FILE: memory_vector_store.py
# LOCATION: C:\Users\Admin\Desktop\ai-agent-system\memory_vector_store.py
# ACTION: Replace entire file

import json
import math
import os
import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL      = "nomic-embed-text"


# ---------------------------------------------------------------------------
# Helper — cosine similarity (pure Python, no numpy required)
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Return the cosine similarity between two equal-length vectors.

    Returns 0.0 if either vector has zero magnitude.
    """
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Helper — embed text via Ollama nomic-embed-text
# ---------------------------------------------------------------------------

def _embed(text: str) -> list:
    """Request an embedding vector from the local Ollama instance.

    Uses nomic-embed-text (the same model as legal_faiss.py) so that
    semantic search across memory entries is in the same embedding space
    as the main legal case vector store.

    Raises RuntimeError if the HTTP call fails or returns no vector.
    """
    payload = {"model": EMBED_MODEL, "prompt": text}
    try:
        response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

    data   = response.json()
    vector = data.get("embedding")
    if not vector:
        raise RuntimeError(
            f"Ollama returned no embedding vector. Response: {data!r}"
        )
    return vector


# ---------------------------------------------------------------------------
# VectorMemoryStore
# ---------------------------------------------------------------------------

class VectorMemoryStore:
    """Semantic memory store backed by a JSON file.

    Uses Ollama nomic-embed-text for embeddings — the same model as the
    main legal_faiss.py retrieval layer — ensuring the two stores operate
    in the same embedding space.

    Replaces the previous implementation which used sentence_transformers
    and all-MiniLM-L6-v2 (a different, incompatible embedding space).

    Improvements:
    1. Embeddings via Ollama nomic-embed-text (compatible with legal_faiss.py).
    2. No numpy dependency — cosine similarity computed in pure Python.
    3. Atomic saves — writes to .tmp then os.replace() to prevent corruption.
    4. Explicit encoding="utf-8" on all file operations.
    5. Safe _load() — returns empty list on JSONDecodeError or OSError.
    """

    def __init__(self, path="data/vector_memory.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if not os.path.exists(self.path):
            self._atomic_save([])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self):
        """Load and return the full entry list from disk.

        Returns an empty list if the file is missing, empty, or corrupt.
        """
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, ValueError):
            return []

    def _atomic_save(self, data):
        """Write data to disk atomically.

        Writes to <path>.tmp first, then calls os.replace() to swap it in.
        This ensures the file is never left in a partially-written state.
        """
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    # ── Public API ────────────────────────────────────────────────────────────

    def add(self, text, metadata=None):
        """Embed text and append the entry to the store, then save atomically."""
        data = self._load()

        try:
            embedding = _embed(text)
        except RuntimeError as exc:
            print(f"[memory_vector_store] Embedding failed — entry not saved: {exc}")
            return

        data.append({
            "text":      text,
            "embedding": embedding,
            "metadata":  metadata or {},
        })

        self._atomic_save(data)

    def search(self, query, top_k=5):
        """Return the top_k most semantically similar entries to query.

        Returns an empty list if the store is empty or embedding fails.
        """
        data = self._load()

        if not data:
            return []

        try:
            query_emb = _embed(query)
        except RuntimeError as exc:
            print(f"[memory_vector_store] Embedding failed during search: {exc}")
            return []

        scored = []
        for item in data:
            emb = item.get("embedding")
            if not emb:
                continue
            score = _cosine_similarity(query_emb, emb)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
