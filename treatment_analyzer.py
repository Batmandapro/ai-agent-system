import json
import os
import re
from llm import llm

# ── CONFIG ────────────────────────────────────────────────────────────────────
DB_PATH = "data/cases_db.json"

POSITIVE_KEYWORDS = [
    "followed", "applied", "approved", "endorsed", "affirmed",
    "adopted", "accepted", "relied on", "consistent with"
]
NEGATIVE_KEYWORDS = [
    "overruled", "not followed", "declined to follow", "doubted",
    "disapproved", "rejected", "departed from", "inconsistent with"
]
NEUTRAL_KEYWORDS = [
    "distinguished", "considered", "noted", "referred to",
    "cited", "discussed", "mentioned", "queried"
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _load_db():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("chunks", [])
    return []

def _extract_case_name(citation: str) -> str:
    return citation.strip().lower()

def _find_citing_chunks(target: str, db: list) -> list:
    target_lower = _extract_case_name(target)
    short_name   = re.sub(r'\[.*?\]', '', target_lower).strip()
    hits = []
    for entry in db:
        # Handle both "chunk" and "text" field names for backward compatibility
        chunk  = entry.get("text", entry.get("chunk", ""))
        source = entry.get("source", "unknown")
        chunk_lower = chunk.lower()
        if target_lower in chunk_lower or short_name in chunk_lower:
            if short_name in source.lower():
                continue
            hits.append({"source": source, "chunk": chunk})
    return hits

def _classify_treatment(chunk: str) -> str:
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
    prompt = f"""You are a Singapore law case citator assistant.
Do not use any markdown formatting. Do not use bold, italic, or headings. Plain text only.

The following passage is from the case "{source}". It mentions the case "{target}".

Passage:
{chunk}

Classify how "{source}" treats "{target}" using ONLY one of these labels:
FOLLOWED, APPLIED, APPROVED, DISTINGUISHED, CONSIDERED, DOUBTED, NOT FOLLOWED, OVERRULED

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

def analyse_treatment(target_case: str) -> dict:
    """
    Find all cases in the database that cite the target case,
    classify their treatment, and return a structured report.
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
            "message":  (
                "No citing cases found in the current database. "
                "Add more cases via ingest.py to improve coverage."
            )
        }

    seen_sources = {}
    for hit in hits:
        src = hit["source"]
        if src not in seen_sources:
            seen_sources[src] = hit

    unique_hits = list(seen_sources.values())
    results = {"positive": [], "neutral": [], "negative": [], "unknown": []}

    for hit in unique_hits:
        classified      = _llm_classify(target_case, hit["source"], hit["chunk"])
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

# Alias for backward compatibility
analyze_treatment = analyse_treatment


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
            lines.append(f"  {item['source']}")
            lines.append(f"    Treatment : {item['treatment']}")
            lines.append(f"    At para   : {item['paragraph']}")
            lines.append(f"    Note      : {item['explanation']}")
            lines.append("")

    lines.append("=" * 60 + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    target = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Low Kok Heng"
    print(f"\nAnalysing treatment of: {target}")
    report = analyse_treatment(target)
    print(format_treatment_report(report))