def route(query: str) -> str:
    q = query.lower()

    # CASE SUMMARY MODE
    if any(x in q for x in [
        "summarise case",
        "summarize case",
        "facts of",
        "holding of",
        "case of"
    ]):
        return "case_summary"

    # SYNTHESIS MODE
    if any(x in q for x in [
        "compare",
        "distinguish",
        "vs",
        "versus",
        "conflict",
        "reconcile"
    ]):
        return "synthesis"

    # DEFAULT = IRAC
    return "irac"