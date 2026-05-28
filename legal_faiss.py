# FILE: legal_faiss.py
# REPLACES: C:\Users\Admin\Desktop\ai-agent-system\legal_faiss.py

"""
legal_faiss.py — Retrieval layer for the Singapore Legal AI system.

Improvements over the original:
1. Similarity threshold  — only returns chunks scoring >= 0.55 (fallback: 0.40).
2. Adaptive k            — k adjusts between 5 and 12 based on top-result confidence.
3. Deduplication         — near-duplicate chunks from the same source are collapsed.
4. Area filter           — optional ``area`` parameter restricts search to a legal folder.
5. Query expansion       — common Singapore legal terms are expanded before embedding.
6. Scores exposed        — every returned chunk carries a ``score`` field (float).
"""

import json
import math
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = Path("data/cases_db.json")
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# Similarity thresholds
PRIMARY_THRESHOLD = 0.55      # Minimum score to include a chunk
FALLBACK_THRESHOLD = 0.40     # Used when fewer than 2 chunks pass the primary threshold
MIN_RESULTS_BEFORE_FALLBACK = 2

# Adaptive k boundaries
DEFAULT_K = 8
HIGH_CONFIDENCE_SCORE = 0.80  # Top result >= this → reduce to k=5 (high precision)
LOW_CONFIDENCE_SCORE = 0.55   # Top result < this  → increase to k=12 (cast wider net)
K_HIGH_CONFIDENCE = 5
K_LOW_CONFIDENCE = 12

# Deduplication — chunks from the same source above this similarity are collapsed
DEDUP_SIMILARITY_THRESHOLD = 0.90

# ---------------------------------------------------------------------------
# Query expansion dictionary (Singapore legal context)
# Keys are lowercase substrings to detect in the query.
# Values are the expansion phrases appended before embedding.
# ---------------------------------------------------------------------------

QUERY_EXPANSION: dict[str, str] = {
    "negligence": "duty of care breach causation damage tort",
    "duty of care": "negligence reasonable foreseeability proximity",
    "occupier": "occupier liability visitor trespasser premises duty",
    "nuisance": "private nuisance public nuisance unreasonable interference",
    "defamation": "libel slander publication false statement reputation",
    "contract": "offer acceptance consideration intention to create legal relations",
    "breach": "breach of contract repudiation damages remedy",
    "misrepresentation": "fraudulent negligent innocent misrepresentation rescission",
    "mens rea": "criminal intent fault element knowledge recklessness",
    "actus reus": "criminal act guilty act omission conduct element",
    "murder": "homicide intention to kill grievous hurt mens rea Penal Code",
    "robbery": "theft force hurt criminal force Penal Code",
    "cheating": "fraud deceit dishonest inducement property Penal Code",
    "judicial review": "administrative law certiorari mandamus quashing order prerogative",
    "injunction": "interlocutory mandatory prohibitory equitable relief balance of convenience",
}


# ---------------------------------------------------------------------------
# Helper — cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Return the cosine similarity between two vectors.

    Returns 0.0 if either vector has zero magnitude to avoid division by zero.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Helper — query expansion
# ---------------------------------------------------------------------------

def _expand_query(query: str) -> str:
    """Append expansion phrases for recognised Singapore legal keywords.

    Matching is case-insensitive and substring-based.  Multiple expansions
    are accumulated and appended once, separated by spaces.
    """
    lower_query = query.lower()
    expansions: list[str] = []
    for keyword, phrase in QUERY_EXPANSION.items():
        if keyword in lower_query and phrase not in expansions:
            expansions.append(phrase)
    if expansions:
        expanded = query + " " + " ".join(expansions)
        logger.debug("Query expanded: %r → %r", query, expanded)
        return expanded
    return query


# ---------------------------------------------------------------------------
# Helper — embed text via Ollama
# ---------------------------------------------------------------------------

def _embed(text: str) -> list[float]:
    """Request an embedding vector from the local Ollama instance.

    Raises ``RuntimeError`` if the HTTP call fails or returns no vector.
    """
    payload = {"model": EMBED_MODEL, "prompt": text}
    try:
        response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

    data = response.json()
    vector = data.get("embedding")
    if not vector:
        raise RuntimeError(
            f"Ollama returned no embedding vector. Response: {data!r}"
        )
    return vector


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LegalFAISS:
    """Retrieval layer for the Singapore Legal AI system.

    Loads the pre-computed case database once at instantiation and performs
    in-memory cosine similarity search on every query.  The class name and
    public method signatures are preserved for compatibility with api.py.

    Usage::

        rag = LegalFAISS()
        results = rag.query("What constitutes negligence?", k=5, area="tort")
    """

    def __init__(self) -> None:
        self._db: list[dict] = self._load_db()
        logger.info("LegalFAISS initialised with %d entries.", len(self._db))

    # ------------------------------------------------------------------
    # Public API (signatures kept compatible with existing callers)
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of entries in the loaded case database.

        Called by the /health endpoint in api.py.
        """
        return len(self._db)

    def query(
        self,
        text: str,
        k: int = 5,         # Signature preserved; adaptive k overrides this internally
        area: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve the most relevant case chunks for a legal query.

        Parameters
        ----------
        text:
            The raw user query string.
        k:
            Baseline for adaptive-k calculation.  The actual number of
            candidates retrieved is determined by the top-result confidence
            score (see ``DEFAULT_K``, ``K_HIGH_CONFIDENCE``, ``K_LOW_CONFIDENCE``).
            Passing k=5 (the legacy default) still works; adaptive k overrides it.
        area:
            Optional legal area string (e.g. ``"criminal"``, ``"tort"``).
            When provided, only DB entries whose ``folder`` field contains
            this string are considered.  Passed by the intent_router.

        Returns
        -------
        list[dict]
            Each element is a copy of the DB entry with an additional
            ``"score"`` key (float, cosine similarity).  Ordered by score
            descending.  After deduplication, at most *adaptive_k* items
            are returned.
        """
        # 1. Query expansion
        expanded_text = _expand_query(text)

        # 2. Embed the (possibly expanded) query
        try:
            query_vector = _embed(expanded_text)
        except RuntimeError as exc:
            logger.error("Embedding failed: %s", exc)
            return []

        # 3. Area filter — restrict candidate pool before scoring
        candidates = self._filter_by_area(self._db, area)
        if not candidates:
            logger.warning(
                "Area filter '%s' matched no entries; falling back to full DB.", area
            )
            candidates = self._db

        # 4. Score all candidates
        scored = self._score_candidates(candidates, query_vector)
        if not scored:
            return []

        # 5. Determine adaptive k
        top_score = scored[0]["score"]
        adaptive_k = self._adaptive_k(top_score)
        logger.debug(
            "Top score: %.4f → adaptive_k=%d (caller requested k=%d)",
            top_score, adaptive_k, k,
        )

        # 6. Apply similarity threshold with fallback
        filtered = self._apply_threshold(scored, adaptive_k)

        # 7. Deduplicate across chunks from the same source
        deduplicated = self._deduplicate(filtered)

        # 8. Trim to adaptive_k after deduplication
        results = deduplicated[:adaptive_k]

        logger.info(
            "query() → %d results (area=%r, adaptive_k=%d, top_score=%.4f)",
            len(results), area, adaptive_k, top_score,
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_db() -> list[dict]:
        """Load and return the case database from disk.

        Raises ``FileNotFoundError`` if the DB file is missing.
        """
        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"Case database not found at '{DB_PATH}'. "
                "Ensure data/cases_db.json is present before starting the server."
            )
        with DB_PATH.open("r", encoding="utf-8") as fh:
            db = json.load(fh)
        logger.info("Loaded %d entries from '%s'.", len(db), DB_PATH)
        return db

    @staticmethod
    def _filter_by_area(db: list[dict], area: Optional[str]) -> list[dict]:
        """Return only entries whose ``folder`` field contains *area*.

        If *area* is ``None`` or an empty string, the full list is returned
        unchanged.  Matching is case-insensitive.
        """
        if not area:
            return db
        area_lower = area.lower()
        return [
            entry for entry in db
            if area_lower in entry.get("folder", "").lower()
        ]

    @staticmethod
    def _score_candidates(
        candidates: list[dict],
        query_vector: list[float],
    ) -> list[dict]:
        """Compute cosine similarity for each candidate and return sorted results.

        Each item in the returned list is a *shallow copy* of the DB entry
        with a ``"score"`` key added.  Results are sorted descending by score.
        """
        scored: list[dict] = []
        for entry in candidates:
            doc_vector = entry.get("vector")
            if not doc_vector:
                continue
            sim = _cosine_similarity(query_vector, doc_vector)
            item = dict(entry)   # Shallow copy — does not mutate the loaded DB
            item["score"] = round(sim, 6)
            scored.append(item)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    @staticmethod
    def _adaptive_k(top_score: float) -> int:
        """Return the appropriate retrieval depth based on top-result confidence.

        >=0.80 → k=5   (high confidence — fewer, more precise results)
        <0.55  → k=12  (low confidence — cast a wider net)
        else   → k=8   (default)
        """
        if top_score >= HIGH_CONFIDENCE_SCORE:
            return K_HIGH_CONFIDENCE
        if top_score < LOW_CONFIDENCE_SCORE:
            return K_LOW_CONFIDENCE
        return DEFAULT_K

    @staticmethod
    def _apply_threshold(
        scored: list[dict],
        adaptive_k: int,
    ) -> list[dict]:
        """Filter results by similarity threshold, with a fallback.

        Primary threshold: 0.55
        If fewer than 2 results pass, retry with fallback threshold: 0.40.
        This ensures the system always returns something meaningful rather
        than an empty context — which itself causes hallucinations.
        """
        primary = [e for e in scored if e["score"] >= PRIMARY_THRESHOLD]
        if len(primary) >= MIN_RESULTS_BEFORE_FALLBACK:
            return primary[:adaptive_k]

        logger.debug(
            "Only %d result(s) passed primary threshold %.2f; "
            "applying fallback threshold %.2f.",
            len(primary), PRIMARY_THRESHOLD, FALLBACK_THRESHOLD,
        )
        fallback = [e for e in scored if e["score"] >= FALLBACK_THRESHOLD]
        return fallback[:adaptive_k]

    @staticmethod
    def _deduplicate(entries: list[dict]) -> list[dict]:
        """Remove near-duplicate chunks that originate from the same source file.

        For each source, if two chunks have cosine similarity > 0.90 with
        each other, only the higher-scoring chunk is retained.  Chunks are
        processed in score-descending order so the best one always survives.

        This prevents the LLM from receiving the same passage multiple times,
        which skews reasoning toward repeated facts.
        """
        kept: list[dict] = []

        for candidate in entries:
            candidate_source = candidate.get("source", "")
            candidate_vector = candidate.get("vector", [])
            is_duplicate = False

            for existing in kept:
                # Only compare chunks from the same source file
                if existing.get("source", "") != candidate_source:
                    continue
                existing_vector = existing.get("vector", [])
                if not existing_vector or not candidate_vector:
                    continue
                sim = _cosine_similarity(candidate_vector, existing_vector)
                if sim > DEDUP_SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    logger.debug(
                        "Deduped chunk from '%s' (sim=%.4f with retained chunk).",
                        candidate_source, sim,
                    )
                    break

            if not is_duplicate:
                kept.append(candidate)

        return kept
