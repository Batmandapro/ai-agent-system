import re
from intent_router import route
from legal_distiller import distil
from reasoning_engine import reason
from formatter import format_response, format_error, format_no_results

# ── CONFIG ────────────────────────────────────────────────────────────────────
BANNER = """
============================================================
  Legal AI — Singapore Criminal Law Assistant
  Type your query and press Enter.
  Commands:
    exit / quit   — Exit
    stats         — Show database statistics
    memory        — Show memory log
    rules         — Show writing style rules
============================================================
"""

# ── TOOL DETECTION ────────────────────────────────────────────────────────────

# Acts and abbreviations to watch for in queries
STATUTE_TRIGGERS = {
    "mda":           ("MDA", None),
    "misuse of drugs": ("MDA", None),
    "penal code":    ("Penal Code", None),
    " pc ":          ("Penal Code", None),
    "cpc":           ("CPC", None),
    "criminal procedure": ("CPC", None),
    "evidence act":  ("Evidence Act", None),
    "arms offences": ("Arms Offences Act", None),
    "women's charter": ("Women's Charter", None),
    "efma":          ("EFMA", None),
    "computer misuse": ("Computer Misuse Act", None),
    "pca":           ("PCA", None),
    "corruption":    ("PCA", None),
}

# Regex to detect section references e.g. "s 5", "s5", "section 5", "s 5A", "s 5(1)"
SECTION_PATTERN = re.compile(
    r'\b(?:s(?:ection)?\.?\s*)(\d+[A-Z]?(?:\(\d+\))?(?:\([a-z]\))?)',
    re.IGNORECASE
)

def _detect_statutes(query: str) -> list:
    """
    Scan the query for statute references and section numbers.
    Returns a list of dicts: [{"act": "MDA", "section": "5"}, ...]
    """
    query_lower = query.lower()
    found_acts  = []

    for trigger, (act_name, _) in STATUTE_TRIGGERS.items():
        if trigger in query_lower:
            # Find section number if present
            sections = SECTION_PATTERN.findall(query)
            if sections:
                for sec in sections:
                    found_acts.append({"act": act_name, "section": sec})
            else:
                found_acts.append({"act": act_name, "section": None})

    # Deduplicate
    seen = set()
    unique = []
    for item in found_acts:
        key = (item["act"], item["section"])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique

def _detect_treatment_request(query: str) -> str:
    """
    Detect if the query asks about how a case has been treated.
    Returns the target case name if detected, else empty string.
    """
    query_lower = query.lower()
    triggers    = [
        "treatment of", "how has", "subsequent history",
        "been followed", "been distinguished", "been overruled",
        "cases citing", "cases that cite", "how is", "treatment"
    ]
    if any(t in query_lower for t in triggers):
        # Try to extract a case name — look for capitalised words or citation pattern
        match = re.search(r'[A-Z][a-z]+(?: [A-Z][a-z]+){1,4}(?:\s*\[\d{4}\])?', query)
        if match:
            return match.group(0).strip()
    return ""

def _detect_research_request(query: str) -> bool:
    """Detect if the query asks for active online research."""
    triggers = [
        "research", "find cases on", "look up cases",
        "search for cases", "find me cases", "what cases are there"
    ]
    return any(t in query.lower() for t in triggers)

# ── TOOL RUNNERS ──────────────────────────────────────────────────────────────

def _run_statute_lookup(statute_refs: list) -> str:
    """Run statute lookups and return formatted results as context."""
    try:
        from statute_lookup_tool import lookup_section
        results = []
        for ref in statute_refs:
            result = lookup_section(ref["act"], ref["section"])
            if result["found"]:
                sec_label = f" s {ref['section']}" if ref["section"] else ""
                results.append(
                    f"[STATUTE] {ref['act']}{sec_label} ({result['cap']}):\n{result['text']}"
                )
        return "\n\n".join(results)
    except Exception as e:
        return ""

def _run_treatment_analysis(target_case: str) -> str:
    """Run treatment analysis and return formatted report."""
    try:
        from treatment_analyzer import analyze_treatment, format_treatment_report
        report = analyze_treatment(target_case)
        return format_treatment_report(report)
    except Exception as e:
        return ""

def _run_online_research(query: str, mode: str) -> dict:
    """Trigger the research agent for active online research."""
    try:
        from research_agent import research
        print("[AUTO] Online research triggered — searching free sources...")
        return {"research_output": research(query, use_online=True, use_lawnet=False)}
    except Exception as e:
        return {}

# ── CORE PIPELINE ─────────────────────────────────────────────────────────────

def run_query(query: str) -> str:
    """
    Full auto-tool pipeline:

    1. Route intent
    2. Auto-detect which tools are needed
    3. Run tools automatically
    4. Retrieve and distil local case law
    5. Combine all context (statute text + case law + tool outputs)
    6. Reason over combined context
    7. Format and return
    """

    mode = route(query)

    # ── Auto Tool Detection ───────────────────────────────────────────────────
    statute_refs     = _detect_statutes(query)
    treatment_target = _detect_treatment_request(query)
    research_needed  = _detect_research_request(query)

    # ── Run Tools Automatically ───────────────────────────────────────────────
    supplementary_context = ""
    appended_output       = ""

    # Tool 1 — Statute lookup
    if statute_refs:
        print(f"[AUTO] Statute reference detected — looking up {[r['act'] for r in statute_refs]}...")
        statute_text = _run_statute_lookup(statute_refs)
        if statute_text:
            supplementary_context += f"\n\n{statute_text}"
            print(f"[AUTO] Statute text retrieved.")

    # Tool 2 — Treatment analysis
    if treatment_target:
        print(f"[AUTO] Treatment query detected — analysing '{treatment_target}'...")
        treatment_output = _run_treatment_analysis(treatment_target)
        if treatment_output:
            appended_output += treatment_output
            print(f"[AUTO] Treatment analysis complete.")

    # Tool 3 — Online research (asks for confirmation first)
    if research_needed:
        confirm = input(f"[AUTO] Research query detected. Search online sources? (y/n): ").strip().lower()
        if confirm in ("y", "yes"):
            research_result = _run_online_research(query, mode)
            if research_result.get("research_output"):
                return research_result["research_output"]

    # ── Local Retrieval ───────────────────────────────────────────────────────
    distilled = distil(query, mode)

    # Inject supplementary statute context into chunks
    if supplementary_context:
        distilled["raw_chunks"].insert(0, {
            "source": "Singapore Statutes Online (AGC)",
            "chunk":  supplementary_context.strip()
        })
        distilled["sources"].insert(0, "Singapore Statutes Online (AGC)")
        distilled["chunk_count"] += 1

    if distilled["chunk_count"] == 0:
        return format_no_results(query)

    # ── Reasoning ─────────────────────────────────────────────────────────────
    raw_answer = reason(
        query=query,
        context_chunks=distilled["raw_chunks"],
        mode=mode
    )

    # ── Formatting ────────────────────────────────────────────────────────────
    response = format_response(
        raw=raw_answer,
        mode=mode,
        sources=distilled["sources"],
        query=query
    )

    # Append treatment analysis if triggered
    if appended_output:
        response += appended_output

    return response

# ── COMMANDS ──────────────────────────────────────────────────────────────────

def handle_stats():
    from legal_faiss import stats
    stats()

def handle_memory():
    try:
        from style_learner import review_memory
        review_memory()
    except Exception:
        print("  Memory log not yet set up. Run: python style_learner.py bootstrap")

def handle_rules():
    try:
        from style_learner import review_rules
        review_rules()
    except Exception:
        print("  Writing rules not yet set up. Run: python style_learner.py bootstrap")

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def chat():
    print(BANNER)

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Exiting. Goodbye.")
            break

        if query.lower() == "stats":
            handle_stats()
            continue

        if query.lower() == "memory":
            handle_memory()
            continue

        if query.lower() == "rules":
            handle_rules()
            continue

        try:
            response = run_query(query)
            print(response)
        except Exception as e:
            print(format_error(f"Unexpected error: {e}"))

if __name__ == "__main__":
    chat()