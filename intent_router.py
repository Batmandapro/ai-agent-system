import re

# ── INTENT ROUTER ─────────────────────────────────────────────────────────────
# Routes user queries to one of 9 modes:
#   irac         — full IRAC legal analysis
#   case_summary — summarise a named case
#   synthesis    — compare / reconcile multiple cases
#   sentencing   — sentencing precedents and ranges
#   elements     — elements of an offence
#   procedure    — criminal procedure and process
#   drafting     — draft legal documents / submissions
#   code_review  — review a project file for bugs and improvements
#   self_improve — propose self-improvement strategies for the system

# ── AREA-OF-LAW DETECTION ────────────────────────────────────────────────────

_AREA_KEYWORDS = {
    "criminal": [
        "charge", "accused", "prosecution", "offence", "penal code",
        "mda", "misuse of drugs", "cpc", "criminal procedure",
        "sentencing", "imprisonment", "conviction", "acquittal",
        "mens rea", "actus reus", "guilty", "plea", "mitigation",
        "district court", "high court criminal", "magistrate", "caning",
        "capital punishment", "death penalty", "bail", "remand",
    ],
    "tort": [
        "negligence", "duty of care", "breach of duty", "tortfeasor",
        "personal injury", "damages", "psychiatric harm", "nervous shock",
        "occupier", "trespass", "nuisance", "defamation", "libel", "slander",
        "vicarious liability", "product liability", "tortious", "tort",
        "donoghue", "caparo", "res ipsa loquitur",
    ],
    "contract": [
        "contract", "agreement", "consideration", "offer", "acceptance",
        "misrepresentation", "breach", "frustration", "termination",
        "specific performance", "injunction contract", "condition",
        "warranty", "indemnity", "exclusion clause", "unfair contract",
        "penalty clause", "liquidated damages", "privity",
    ],
    "equity": [
        "trust", "fiduciary", "beneficial interest", "trustee",
        "beneficiary", "constructive trust", "resulting trust",
        "unjust enrichment", "proprietary estoppel", "tracing",
        "breach of fiduciary", "secret profit", "account of profits",
        "equitable", "injunction", "mareva", "anton piller",
    ],
    "admin": [
        "judicial review", "administrative law", "certiorari", "mandamus",
        "prohibition", "public law", "legitimate expectation",
        "wednesbury", "irrationality", "proportionality",
        "natural justice", "fair hearing", "bias", "rule of law",
        "minister", "public authority", "statutory power",
    ],
    "family": [
        "divorce", "ancillary matters", "custody", "care and control",
        "access", "maintenance", "matrimonial", "matrimonial home",
        "women's charter", "family court", "children", "adoption",
        "prenuptial", "postnuptial", "family violence",
    ],
}

def detect_area_of_law(query: str) -> str:
    """
    Detect the area of law from a query.
    Returns one of: criminal | tort | contract | equity | admin | family | general
    """
    q = query.lower()
    scores = {area: 0 for area in _AREA_KEYWORDS}
    for area, keywords in _AREA_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[area] += 1
    best_area  = max(scores, key=scores.get)
    best_score = scores[best_area]
    if best_score == 0:
        return "general"
    return best_area

# ── PRIMARY MODE ROUTER ────────────────────────────────────────────────────────

def route(query: str) -> str:
    """Route a user query to a processing mode."""
    q = query.lower().strip()

    # ── CODE REVIEW ───────────────────────────────────────────────────────────
    if any(x in q for x in [
        "review my code", "review the code", "check my code",
        "audit my code", "read my code", "code review",
        "review legal_faiss", "review ingest", "review app",
        "review intent_router", "review reasoning_engine",
        "review formatter", "review treatment_analyzer",
        "review commonlii_tool", "review elitigation_tool",
        "review research_agent", "list files", "show files",
        "improvement log", "past improvements",
        "rewrite legal_faiss", "rewrite ingest", "rewrite app",
        "rewrite intent_router", "rewrite reasoning_engine",
    ]):
        return "code_review"

    # ── SELF-IMPROVEMENT ──────────────────────────────────────────────────────
    if any(x in q for x in [
        "improve yourself", "improve the system", "how can you improve",
        "how can we improve", "self improve", "self-improve",
        "suggest improvements", "what improvements", "how to improve",
        "improve retrieval", "improve accuracy", "improve speed",
        "improve the llm", "improve embeddings", "improve ingestion",
        "what should we fix", "what needs fixing", "what is wrong with",
        "propose changes", "next steps for the code",
    ]):
        return "self_improve"

    # ── CASE SUMMARY ──────────────────────────────────────────────────────────
    if any(x in q for x in [
        "summarise", "summarize", "summary of",
        "facts of", "holding of", "decision in",
        "what happened in", "case of", "tell me about"
    ]):
        return "case_summary"

    # ── SYNTHESIS ─────────────────────────────────────────────────────────────
    if any(x in q for x in [
        "compare", "distinguish", "vs", "versus",
        "conflict between", "reconcile", "difference between",
        "consistent with", "inconsistent with", "how does"
    ]):
        return "synthesis"

    # ── SENTENCING ────────────────────────────────────────────────────────────
    if any(x in q for x in [
        "sentence", "sentencing", "imprisonment", "fine",
        "custodial", "tariff", "starting point", "benchmark",
        "how much jail", "how long", "penalty", "punishment",
        "mandatory minimum", "caning", "strokes"
    ]):
        return "sentencing"

    # ── ELEMENTS OF OFFENCE ───────────────────────────────────────────────────
    if any(x in q for x in [
        "elements of", "ingredients of", "constitute",
        "what makes", "actus reus", "mens rea",
        "guilty of", "liable for", "offence of",
        "what is needed", "requirements for"
    ]):
        return "elements"

    # ── PROCEDURE ─────────────────────────────────────────────────────────────
    if any(x in q for x in [
        "procedure", "process", "steps", "how to",
        "application", "file", "filing", "court",
        "magistrate", "district court", "high court",
        "criminal procedure code", "cpc", "arrest",
        "bail", "charge", "plead", "plea", "trial",
        "appeal", "revision", "mention"
    ]):
        return "procedure"

    # ── DRAFTING ──────────────────────────────────────────────────────────────
    if any(x in q for x in [
        "draft", "write", "prepare", "letter",
        "submission", "mitigation", "plea in mitigation",
        "written representation", "skeletal", "argument",
        "memorial", "document"
    ]):
        return "drafting"

    # ── AREA-OF-LAW HOOK ──────────────────────────────────────────────────────
    area = detect_area_of_law(query)
    # Future: return f"irac_{area}" once area-specific modes are implemented

    # ── DEFAULT: IRAC ─────────────────────────────────────────────────────────
    return "irac"