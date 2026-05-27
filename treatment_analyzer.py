import json
import os
import re
from llm import llm

# ── CONFIG ────────────────────────────────────────────────────────────────────
DB_PATH = "data/cases_db.json"

# Treatment classification keywords
POSITIVE_KEYWORDS  = ["followed", "applied", "approved", "endorsed", "affirmed",
                       "adopted", "accepted", "relied on", "consistent with"]
NEGATIVE_KEYWORDS  = ["overruled", "not followed", "declined to follow", "doubted",
                       "disapproved", "rejected", "departed from", "inconsistent with"]
NEUTRAL_KEYWORDS   = ["distinguished", "considered", "noted", "referred to",
                       "cited", "discussed", "mentioned", "queried"]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _load_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r") as f:
        return json.load(f)

def _extract_case_name(citation: str) -> str:
    """
    Normalise a citation into searchable variants.
    e.g. 'Low Kok Heng' → ['low kok heng', 'low kok heng [2007]']
    """
    return citation.strip().lower()

def _find_citing_chunks(target: str, db: list) -> list:
    """
    Search all chunks for any that mention the target case.
    Returns list of dicts with source, chunk text, and raw treatment hint.
    """
    target_lower = _extract_case_name(target)
    # Also try just the party name without citation year
    short_name   = re.sub(r'\[.*?\]', '', target_lower).strip()

    hits = []
    for entry in db:
        chunk  = entry.get("chunk", "")
        source = entry.get("source", "unknown")
        chunk_lower = chunk.lower()

        if target_lower in chunk_lower or short_name in chunk_lower:
            # Skip if this chunk is FROM the target case itself
            if short_name in source.lower():
                continue
            hits.append({
                "source": source,
                "chunk":  chunk
            })
    return hits

def _classify_treatment(chunk: str) -> str:
    """
    Quick keyword-based classification before sending to LLM.
    Returns 'positive', 'negative', 'neutral', or 'unknown'.
    """
    chunk_lower = chunk.lower()
    for kw in NEGATIVE_KEYWORDS:
        if kw in chunk_lower:
            return "negative"
    for kw in POSITIVE_KEYWORDS:
        if kw in chunk_lower:
            return "positive"
    for kw in NEUTRAL_KEYWORDS:
        if kw in chunk_lower:
            return "neutral"
    return "unknown"

def _llm_classify(target: str, source: str, chunk: str) -> dict:
    """
    Use LLM to produce a precise treatment classification and explanation.
    """
    prompt = f"""You are a Singapore law case citator assistant.

The following passage is from the case "{source}". It mentions the case "{target}".

Passage:
{chunk}

Classify how "{source}" treats "{target}" using ONLY one of these labels:
- FOLLOWED — the later court followed and applied the earlier case
- APPLIED — the earlier case's principle was applied to the facts
- APPROVED — the earlier case was expressly approved
- DISTINGUISHED — the earlier case was distinguished on the facts or law
- CONSIDERED — the earlier case was merely discussed or noted
- DOUBTED — the earlier case's correctness was questioned
- NOT FOLLOWED — the later court declined to follow the earlier case
- OVERRULED — the earlier case was expressly overruled

Then provide:
1. The treatment label (one of the above)
2. A one-sentence explanation quoting or paraphrasing the relevant part of the passage
3. The paragraph number if visible in the passage (e.g. [23]), or "not identified"

Respond in this exact format:
Treatment: [LABEL]
Explanation: [one sentence]
Paragraph: [number or "not identified"]
"""
    response = llm.invoke(prompt).content.strip()

    # Parse response
    treatment   = "CONSIDERED"
    explanation = ""
    paragraph   = "not identified"

    for line in response.splitlines():
        if line.startswith("Treatment:"):
            treatment = line.replace("Treatment:", "").strip()
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "").strip()
        elif line.startswith("Paragraph:"):
            paragraph = line.replace("Paragraph:", "").strip()

    return {
        "source":      source,
        "treatment":   treatment,
        "explanation": explanation,
        "paragraph":   paragraph,
        "chunk":       chunk
    }

# ── PUBLIC API ────────────────────────────────────────────────────────────────

def analyze_treatment(target_case: str) -> dict:
    """
    Find all cases in the database that cite the target case,
    classify their treatment, and return a structured report.

    Args:
        target_case: Case name/citation e.g. "PP v Low Kok Heng [2007] 4 SLR(R) 183"

    Returns:
        {
            "target":   case name,
            "positive": [...],
            "neutral":  [...],
            "negative": [...],
            "unknown":  [...],
            "total":    int
        }
    """
    db   = _load_db()
    hits = _find_citing_chunks(target_case, db)

    if not hits:
        return {
            "target":   target_case,
            "positive": [],
            "neutral":  [],
            "negative": [],
            "unknown":  [],
            "total":    0,
            "message":  "No citing cases found in the current database. Add more cases via ingest.py to improve coverage."
        }

    # Deduplicate by source — take the most relevant chunk per source
    seen_sources = {}
    for hit in hits:
        src = hit["source"]
        if src not in seen_sources:
            seen_sources[src] = hit

    unique_hits = list(seen_sources.values())

    # Classify each
    results = {"positive": [], "neutral": [], "negative": [], "unknown": []}

    for hit in unique_hits:
        quick = _classify_treatment(hit["chunk"])
        classified = _llm_classify(target_case, hit["source"], hit["chunk"])

        treatment_label = classified["treatment"].upper()

        if any(t in treatment_label for t in ["FOLLOWED", "APPLIED", "APPROVED"]):
            bucket = "positive"
        elif any(t in treatment_label for t in ["OVERRULED", "NOT FOLLOWED", "DOUBTED"]):
            bucket = "negative"
        elif any(t in treatment_label for t in ["DISTINGUISHED", "CONSIDERED"]):
            bucket = "neutral"
        else:
            bucket = "unknown"

        results[bucket].append(classified)

    return {
        "target":   target_case,
        "positive": results["positive"],
        "neutral":  results["neutral"],
        "negative": results["negative"],
        "unknown":  results["unknown"],
        "total":    len(unique_hits)
    }

def format_treatment_report(report: dict) -> str:
    """Format the treatment analysis as a readable terminal report."""

    target = report["target"]
    total  = report["total"]

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("  Case Treatment Analysis")
    lines.append(f"  Target: {target}")
    lines.append("=" * 60)

    if total == 0:
        lines.append(f"\n  {report.get('message', 'No citing cases found.')}")
        lines.append("=" * 60 + "\n")
        return "\n".join(lines)

    lines.append(f"\n  {total} citing case(s) found in database\n")

    sections = [
        ("POSITIVE  (Followed / Applied / Approved)", report["positive"], "+"),
        ("NEUTRAL   (Distinguished / Considered)",    report["neutral"],  "~"),
        ("NEGATIVE  (Doubted / Not Followed / Overruled)", report["negative"], "!"),
        ("UNCLASSIFIED",                              report["unknown"],  "?"),
    ]

    for heading, items, symbol in sections:
        if not items:
            continue
        lines.append(f"  [{symbol}] {heading}")
        lines.append("  " + "-" * 56)
        for item in items:
            lines.append(f"  • {item['source']}")
            lines.append(f"    Treatment : {item['treatment']}")
            lines.append(f"    At para   : {item['paragraph']}")
            lines.append(f"    Note      : {item['explanation']}")
            lines.append("")

    lines.append("=" * 60 + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target = " ".join(sys.argv[1:])
    else:
        target = "Low Kok Heng"

    print(f"\nAnalysing treatment of: {target}")
    report = analyze_treatment(target)
    print(format_treatment_report(report))