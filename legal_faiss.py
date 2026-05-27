import os
import json
import requests
import math

# ── CONFIG ──────────────────────────────────────────────────────────────────
DB_PATH     = "data/cases_db.json"
OLLAMA_URL  = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
TOP_K       = 5  # number of results to return per query

# ── HELPERS ─────────────────────────────────────────────────────────────────
def _load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return []

def _save_db(db):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

def _embed(text):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
    except Exception as e:
        print(f"[FAISS] Embed error: {e}")
        return None

def _cosine_similarity(a, b):
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# ── PUBLIC API ───────────────────────────────────────────────────────────────
def add(source, chunk, vector, folder=""):
    """Add a single chunk+vector to the database."""
    db = _load_db()
    db.append({
        "source": source,
        "folder": folder,
        "chunk":  chunk,
        "vector": vector
    })
    _save_db(db)

def search(query, top_k=TOP_K):
    """
    Embed the query and return the top_k most similar chunks.

    Returns a list of dicts:
    [
        {
            "source":     "filename.pdf",
            "folder":     "data/cases",
            "chunk":      "...text...",
            "score":      0.91
        },
        ...
    ]
    """
    db = _load_db()
    if not db:
        print("[FAISS] Database is empty — run ingest.py first.")
        return []

    query_vector = _embed(query)
    if query_vector is None:
        print("[FAISS] Could not embed query — is Ollama running?")
        return []

    scored = []
    for entry in db:
        vector = entry.get("vector")
        if not vector:
            continue
        score = _cosine_similarity(query_vector, vector)
        scored.append({
            "source": entry.get("source", "unknown"),
            "folder": entry.get("folder", ""),
            "chunk":  entry.get("chunk", ""),
            "score":  round(score, 4)
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def stats():
    """Print a quick summary of the database."""
    db = _load_db()
    if not db:
        print("[FAISS] Database is empty.")
        return

    sources = {}
    for entry in db:
        src = entry.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print(f"[FAISS] Total vectors : {len(db)}")
    print(f"[FAISS] Unique files  : {len(sources)}")
    for src, count in sorted(sources.items()):
        print(f"  {src} — {count} chunks")

if __name__ == "__main__":
    # Quick test — run python legal_faiss.py to verify retrieval is working
    print("Running retrieval test...\n")
    results = search("sentencing for drug trafficking")
    if results:
        for i, r in enumerate(results, 1):
            print(f"Result {i}")
            print(f"  Source : {r['source']}")
            print(f"  Score  : {r['score']}")
            print(f"  Text   : {r['chunk'][:200]}...")
            print()
    else:
        print("No results returned.")