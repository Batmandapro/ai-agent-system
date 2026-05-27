import re
from legal_faiss import LegalFAISS

# ── CONFIG ────────────────────────────────────────────────────────────────────
MIN_SCORE       = 0.10
MAX_CHUNK_CHARS = 1200

_rag = LegalFAISS()

# ── CASE NAME EXTRACTION ──────────────────────────────────────────────────────

_CASE_TRIGGER_PHRASES = [
    "summarise the case of",
    "summarize the case of",
    "summary of the case of",
    "facts of",
    "holding of",
    "decision in",
    "what happened in",
    "tell me about the case",
    "case of",
    "summarise",
    "summarize",
]

def extract_case_name(query: str) -> str | None:
    """
    Attempt to extract a specific case name from the user query.
    Returns lowercase string for source-filename matching, or None.
    """
    q = query.strip()
    q_lower = q.lower()

    triggers = sorted(_CASE_TRIGGER_PHRASES, key=len, reverse=True)

    remainder = None
    for trigger in triggers:
        if trigger in q_lower:
            idx = q_lower.index(trigger) + len(trigger)
            remainder = q[idx:].strip()
            break

    if remainder is None:
        return None

    remainder = re.sub(r'^(?:the|a|an)\s+', '', remainder, flags=re.IGNORECASE).strip()
    remainder = re.sub(r'\[?\d{4}\]?.*$', '', remainder).strip()
    remainder = remainder.strip(".,;:")

    words = [w for w in remainder.split() if len(w) >= 2]
    if len(words) < 2:
        return None

    return remainder.lower()

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
    Accepts raw_chunks already retrieved from LegalFAISS.
    """
    filtered = [c for c in raw_chunks if c.get("score", 0) >= MIN_SCORE]
    unique   = _deduplicate(filtered)
    for c in unique:
        c["text"] = _truncate(c.get("text", ""))
    return unique[:top_k]