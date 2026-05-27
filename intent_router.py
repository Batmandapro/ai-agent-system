def route(query: str) -> str:
    """
    Analyses the query and returns the appropriate reasoning mode.

    Modes:
    - case_summary    : User wants facts/holdings of a specific case
    - synthesis       : User wants comparison or reconciliation across cases
    - sentencing      : User wants sentencing analysis or benchmarks
    - elements        : User wants the legal elements of an offence
    - procedure       : User wants procedural/evidential guidance
    - drafting        : User wants help drafting submissions or mitigation
    - definition      : User wants a legal definition or explanation
    - irac            : Default — apply law to facts in IRAC structure
    """

    q = query.lower()

    # ── CASE SUMMARY ────────────────────────────────────────────────────────
    if any(x in q for x in [
        "summarise case", "summarize case",
        "facts of", "holding of", "case of",
        "what happened in", "decision in",
        "ratio of", "ratio decidendi",
        "obiter in", "obiter dictum"
    ]):
        return "case_summary"

    # ── SYNTHESIS / COMPARISON ───────────────────────────────────────────────
    if any(x in q for x in [
        "compare", "distinguish", "vs", "versus",
        "conflict", "reconcile", "difference between",
        "how does", "contrast", "similarities",
        "consistent with", "inconsistent with"
    ]):
        return "synthesis"

    # ── SENTENCING ───────────────────────────────────────────────────────────
    if any(x in q for x in [
        "sentence", "sentencing", "benchmark",
        "starting point", "imprisonment",
        "fine", "caning", "disqualification",
        "mitigating", "aggravating", "tariff",
        "prevailing sentence", "what is the punishment",
        "how much jail", "how many strokes"
    ]):
        return "sentencing"

    # ── ELEMENTS OF OFFENCE ──────────────────────────────────────────────────
    if any(x in q for x in [
        "elements of", "ingredients of",
        "what must be proven", "what must the prosecution prove",
        "actus reus", "mens rea",
        "constitute", "definition of the offence",
        "what is needed to establish", "establish liability"
    ]):
        return "elements"

    # ── PROCEDURE / EVIDENCE ─────────────────────────────────────────────────
    if any(x in q for x in [
        "procedure", "procedural", "evidence",
        "admissible", "admissibility", "burden of proof",
        "standard of proof", "beyond reasonable doubt",
        "hearsay", "confession", "voir dire",
        "prosecution must", "defence must",
        "how to apply", "how do i file", "what is the process"
    ]):
        return "procedure"

    # ── DRAFTING ─────────────────────────────────────────────────────────────
    if any(x in q for x in [
        "draft", "write", "prepare",
        "mitigation plea", "plea in mitigation",
        "submission", "submissions",
        "letter", "skeletal arguments",
        "help me argue", "how should i frame"
    ]):
        return "drafting"

    # ── DEFINITION / EXPLANATION ─────────────────────────────────────────────
    if any(x in q for x in [
        "what is", "what are", "define",
        "explain", "meaning of", "definition of",
        "what does", "what do you mean"
    ]):
        return "definition"

    # ── DEFAULT: IRAC ────────────────────────────────────────────────────────
    return "irac"