# FILE: statute_lookup_tool.py
# LOCATION: C:\Users\Admin\Desktop\ai-agent-system\statute_lookup_tool.py
# ACTION: Replace entire file

import re

# ── KNOWN STATUTES ─────────────────────────────────────────────────────────────

KNOWN_STATUTES = {
    "penal code":                   ("Penal Code",                    "Cap 224"),
    "pc":                           ("Penal Code",                    "Cap 224"),
    "misuse of drugs act":          ("Misuse of Drugs Act",           "Cap 185"),
    "mda":                          ("Misuse of Drugs Act",           "Cap 185"),
    "criminal procedure code":      ("Criminal Procedure Code",       "Cap 68"),
    "cpc":                          ("Criminal Procedure Code",       "Cap 68"),
    "evidence act":                 ("Evidence Act",                  "Cap 97"),
    "prevention of corruption act": ("Prevention of Corruption Act",  "Cap 241"),
    "pca":                          ("Prevention of Corruption Act",  "Cap 241"),
    "arms offences act":            ("Arms Offences Act",             "Cap 14"),
    "computer misuse act":          ("Computer Misuse Act",           "Cap 50A"),
    "cma":                          ("Computer Misuse Act",           "Cap 50A"),
    "money laundering":             ("Corruption, Drug Trafficking and Other Serious Crimes (Confiscation of Benefits) Act", "Cap 65A"),
    "cdsa":                         ("Corruption, Drug Trafficking and Other Serious Crimes (Confiscation of Benefits) Act", "Cap 65A"),
    "legal profession act":         ("Legal Profession Act",          "Cap 161"),
    "companies act":                ("Companies Act",                  "Cap 50"),
    "employment act":               ("Employment Act",                 "Cap 91"),
    "income tax act":               ("Income Tax Act",                 "Cap 134"),
    "bankruptcy act":               ("Bankruptcy Act",                 "Cap 20"),
    "civil law act":                ("Civil Law Act",                  "Cap 43"),
    "conveyancing act":             ("Conveyancing and Law of Property Act", "Cap 61"),
    "land titles act":              ("Land Titles Act",               "Cap 157"),
}

# ── SECTION PATTERN ────────────────────────────────────────────────────────────

_SECTION_RE = re.compile(
    r'\b(?:s(?:ection)?\s*)(\d+[A-Za-z]?(?:\s*\(\s*\d+\s*\))?)',
    re.IGNORECASE
)

# ── INTERNAL HELPERS ───────────────────────────────────────────────────────────

def _normalise_act_name(text: str) -> str:
    """Lower-case and strip punctuation for lookup key matching."""
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


def _detect_act(query: str):
    """
    Return (full_name, cap_number) if the query references a known statute,
    or (None, None) if no match is found.
    """
    normalised = _normalise_act_name(query)
    for key, value in KNOWN_STATUTES.items():
        if key in normalised:
            return value
    return None, None


def _detect_section(query: str) -> str | None:
    """
    Extract the first section reference from the query string,
    e.g. 'section 5(1)' → '5(1)', 's 300' → '300'.
    Returns None if no section is mentioned.
    """
    m = _SECTION_RE.search(query)
    if m:
        return m.group(1).strip().replace(" ", "")
    return None


def format_statute_result(act_name: str, cap: str, section: str | None) -> str:
    """
    Format a statute lookup result as a plain-text string suitable for
    terminal or API response display.
    """
    if section:
        return (
            f"Statute Reference\n"
            f"  Act     : {act_name} ({cap})\n"
            f"  Section : s {section}\n"
            f"  Note    : Verify the exact text of s {section} against the current "
            f"Singapore Statutes Online version at https://sso.agc.gov.sg"
        )
    return (
        f"Statute Reference\n"
        f"  Act  : {act_name} ({cap})\n"
        f"  Note : Verify the current version at https://sso.agc.gov.sg"
    )


def lookup_section(act_name: str, section: str | None = None) -> str:
    """
    Look up a statute by its canonical act name and optional section number.
    Returns a formatted reference string, or an empty string if the act
    is not in the known statutes dictionary.
    """
    normalised = _normalise_act_name(act_name)
    for key, (full_name, cap) in KNOWN_STATUTES.items():
        if key in normalised or _normalise_act_name(full_name) == normalised:
            return format_statute_result(full_name, cap, section)
    return ""


def lookup_multiple(queries: list) -> list:
    """
    Run lookup_statute over a list of query strings and return a list
    of non-empty result strings.
    """
    results = []
    for q in queries:
        result = lookup_statute(q)
        if result:
            results.append(result)
    return results


def search_statutes(keyword: str) -> list:
    """
    Return a list of (full_name, cap_number) tuples for all known statutes
    whose name or abbreviation contains the given keyword.
    """
    kw = keyword.lower().strip()
    seen = set()
    results = []
    for key, (full_name, cap) in KNOWN_STATUTES.items():
        if kw in key or kw in full_name.lower():
            if full_name not in seen:
                seen.add(full_name)
                results.append((full_name, cap))
    return results


# ── PUBLIC ENTRY POINT ─────────────────────────────────────────────────────────

def lookup_statute(query: str) -> str:
    """
    Accept a free-text query (e.g. 'section 5(1) MDA' or 'Penal Code s 300'),
    detect the referenced act and section, and return a formatted reference string.

    This is the primary entry point called by api.py and app.py.
    It delegates to lookup_section() internally.

    Returns an empty string if no known statute is detected in the query.
    """
    act_name, cap = _detect_act(query)
    if not act_name:
        return ""
    section = _detect_section(query)
    return format_statute_result(act_name, cap, section)


# ── ALIAS ─────────────────────────────────────────────────────────────────────
# lookup_statute is the primary function. The alias below is kept for any
# future code that may call lookup_section(query) with a free-text argument.
lookup_statute_alias = lookup_statute


if __name__ == "__main__":
    import sys
    test_queries = [
        "What is the punishment under section 5(1) MDA?",
        "Penal Code s 300 murder",
        "Is this a CPC section 228 issue?",
        "Companies Act director's duty",
    ]
    queries = sys.argv[1:] if len(sys.argv) > 1 else test_queries
    for q in queries:
        result = lookup_statute(q)
        print(f"\nQuery : {q}")
        print(result if result else "  (no known statute detected)")