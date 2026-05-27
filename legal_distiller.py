from legal_faiss import LegalFAISS

# ── CONFIG ────────────────────────────────────────────────────────────────────
MIN_SCORE       = 0.10  # discard chunks below this cosine similarity score
MAX_CHUNK_CHARS = 1200  # truncate individual chunks if too long

_rag = LegalFAISS()

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _truncate(text, max_chars=MAX_CHUNK_CHARS):
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."

def _deduplicate(chunks):
    """Remove near-duplicate chunks by checking for 80%+ word overlap."""
    seen   = []
    unique = []
    for chunk in chunks:
        text         = chunk.get("text", "")
        is_duplicate = False
        for s in seen:
            overlap = len(set(text.split()) & set(s.split()))
            shorter = min(len(text.split()), len(s.split()))
            if shorter > 0 and overlap / shorter > 0.8:
                is_duplicate = True
                break
        if not is_duplicate:
            seen.append(text)
            unique.append(chunk)
    return unique

# ── PUBLIC API ────────────────────────────────────────────────────────────────

def distil(query: str, raw_chunks: list, top_k: int = 6) -> list:
    """
    Filter, deduplicate and return the most relevant chunks.

    Accepts raw_chunks already retrieved from LegalFAISS.search()
    so that app.py controls retrieval and distiller controls quality.

    Returns a list of chunk dicts ready for reasoning_engine.
    """
    # Step 1 — Filter by minimum score
    filtered = [c for c in raw_chunks if c.get("score", 0) >= MIN_SCORE]

    # Step 2 — Deduplicate
    unique = _deduplicate(filtered)

    # Step 3 — Truncate long chunks
    for c in unique:
        c["text"] = _truncate(c.get("text", ""))

    # Step 4 — Return top_k
    return unique[:top_k]


def distil_full(query: str, mode: str = "irac", top_k: int = 6) -> dict:
    """
    All-in-one version: retrieves from FAISS, filters, deduplicates.
    Returns a dict with context string and metadata.
    Useful for standalone testing.
    """
    raw      = _rag.search(query, top_k=top_k * 2)
    chunks   = distil(query, raw, top_k=top_k)
    sources  = list(dict.fromkeys(
        c.get("source") or c.get("meta", {}).get("source", "Unknown")
        for c in chunks
    ))

    context_lines = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source") or c.get("meta", {}).get("source", "Unknown")
        score  = round(c.get("score", 0), 3)
        text   = c.get("text", "")
        context_lines.append(f"[{i}] SOURCE: {source} (relevance: {score})\n{text}\n")

    return {
        "query":       query,
        "mode":        mode,
        "context":     "\n".join(context_lines),
        "sources":     sources,
        "chunk_count": len(chunks),
        "raw_chunks":  chunks
    }


if __name__ == "__main__":
    print("Testing legal_distiller...\n")
    result = distil_full("sentencing for drug trafficking", mode="sentencing")
    print(f"Query  : {result['query']}")
    print(f"Mode   : {result['mode']}")
    print(f"Chunks : {result['chunk_count']}")
    print(f"Sources: {', '.join(result['sources'])}")
    print(f"\nContext preview:\n{result['context'][:500]}...")