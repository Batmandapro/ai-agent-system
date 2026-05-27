import re

# ── INTENT ROUTER ─────────────────────────────────────────────────────────────
# Routes user queries to one of 7 modes:
#   irac         — full IRAC legal analysis
#   case_summary — summarise a named case
#   synthesis    — compare / reconcile multiple cases
#   sentencing   — sentencing precedents and ranges
#   elements     — elements of an offence
#   procedure    — criminal procedure and process
#   drafting     — draft legal documents / submissions

def route(query: str) -> str:
    q = query.lower().strip()

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

    # ── DEFAULT: IRAC ─────────────────────────────────────────────────────────
    return "irac"