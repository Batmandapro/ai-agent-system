import re
import requests
from urllib.parse import quote

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Singapore Statutes Online — free, authoritative, no authentication required
AGC_BASE      = "https://sso.agc.gov.sg"
AGC_SEARCH    = "https://sso.agc.gov.sg/Browse/Act/Current/All"
REQUEST_DELAY = 1
HEADERS       = {
    "User-Agent": "Mozilla/5.0 (Legal Research Assistant; educational use)"
}

# ── COMMON SINGAPORE STATUTES (fast lookup without needing to search) ─────────
# Maps common short names and abbreviations to their AGC Act IDs
KNOWN_STATUTES = {
    # Criminal Law
    "penal code":                   "Cap 224",
    "pc":                           "Cap 224",
    "criminal procedure code":      "Cap 68",
    "cpc":                          "Cap 68",
    "misuse of drugs act":          "Cap 185",
    "mda":                          "Cap 185",
    "arms offences act":            "Cap 14",
    "aoa":                          "Cap 14",
    "kidnapping act":               "Cap 151",
    "corruption prevention":        "Cap 241",
    "pca":                          "Cap 241",
    "prevention of corruption act": "Cap 241",
    "internal security act":        "Cap 143",
    "isa":                          "Cap 143",
    "vandalism act":                "Cap 341",
    "women's charter":              "Cap 353",
    "employment of foreign manpower act": "Cap 91A",
    "efma":                         "Cap 91A",

    # Evidence & Courts
    "evidence act":                 "Cap 97",
    "supreme court of judicature act": "Cap 322",
    "subordinate courts act":       "Cap 321",
    "state courts act":             "Cap 321",
    "legal profession act":         "Cap 161",

    # Regulatory
    "computer misuse act":          "Cap 50A",
    "cma":                          "Cap 50A",
    "customs act":                  "Cap 70",
    "income tax act":               "Cap 134",
    "companies act":                "Cap 50",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    """Strip HTML tags and clean whitespace."""
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'\s{3,}', '\n\n', text)
    return text.strip()

def _normalise_act_name(name: str) -> str:
    """Normalise an act name for lookup against KNOWN_STATUTES."""
    return name.lower().strip().rstrip(".")

def _fetch_url(url: str) -> str:
    """Fetch a URL and return the page text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[STATUTE] Fetch failed: {e}")
        return ""

def _extract_section(full_text: str, section_number: str) -> str:
    """
    Extract a specific section from statute full text.
    Handles formats like "5", "5A", "5(1)", "5(1)(a)".
    """
    # Try to find the section heading
    patterns = [
        rf'(?:^|\n)\s*{re.escape(section_number)}\s*\.?\s+(.*?)(?=\n\s*\d+[A-Z]?\s*\.|\Z)',
        rf'Section\s+{re.escape(section_number)}\s*[.\-—]\s*(.*?)(?=Section\s+\d|\Z)',
    ]

    for pattern in patterns:
        match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(0).strip()
            # Limit to reasonable length
            if len(text) > 3000:
                text = text[:3000] + "...[truncated]"
            return text

    return ""

# ── PUBLIC API ────────────────────────────────────────────────────────────────

def lookup_section(act_name: str, section: str = None) -> dict:
    """
    Look up a statute or specific section from Singapore Statutes Online.

    Args:
        act_name: Name or abbreviation of the Act e.g. "MDA", "Penal Code", "CPC"
        section:  Optional section number e.g. "5", "5A", "300(a)"

    Returns:
        {
            "act":      full act name,
            "cap":      chapter number,
            "section":  section number or None,
            "text":     extracted text,
            "url":      source URL,
            "found":    bool
        }
    """
    normalised = _normalise_act_name(act_name)
    cap        = KNOWN_STATUTES.get(normalised)

    # Try partial match if exact match fails
    if not cap:
        for key, value in KNOWN_STATUTES.items():
            if normalised in key or key in normalised:
                cap = value
                break

    if not cap:
        return {
            "act":     act_name,
            "cap":     None,
            "section": section,
            "text":    f"Act '{act_name}' not found in known statutes index. Try the full name.",
            "url":     AGC_SEARCH,
            "found":   False
        }

    # Build AGC URL
    cap_clean = cap.replace(" ", "_").replace("/", "_")
    url       = f"{AGC_BASE}/Act/{cap_clean}"

    print(f"[STATUTE] Looking up: {act_name} {cap} s {section or '(full act)'}")
    print(f"[STATUTE] URL: {url}")

    html = _fetch_url(url)
    if not html:
        return {
            "act":     act_name,
            "cap":     cap,
            "section": section,
            "text":    "Could not retrieve statute text from AGC. Check your internet connection.",
            "url":     url,
            "found":   False
        }

    full_text = _strip_html(html)

    # Extract specific section if requested
    if section:
        section_text = _extract_section(full_text, section)
        if not section_text:
            # Return a meaningful snippet around the section number
            idx = full_text.find(f" {section} ")
            if idx > 0:
                section_text = full_text[max(0, idx-100):idx+1500].strip()
            else:
                section_text = f"Section {section} not found in extracted text. Visit {url} directly."
        text = section_text
    else:
        # Return first 3000 chars of the act as a summary
        text = full_text[:3000] + ("...[truncated — specify a section for full text]" if len(full_text) > 3000 else "")

    return {
        "act":     act_name,
        "cap":     cap,
        "section": section,
        "text":    text,
        "url":     url,
        "found":   True
    }


def lookup_multiple(references: list) -> list:
    """
    Look up multiple statute references at once.

    Args:
        references: List of dicts e.g.
                    [{"act": "MDA", "section": "5"},
                     {"act": "Penal Code", "section": "34"}]

    Returns:
        List of lookup result dicts
    """
    results = []
    for ref in references:
        act     = ref.get("act", "")
        section = ref.get("section")
        result  = lookup_section(act, section)
        results.append(result)
    return results


def format_statute_result(result: dict) -> str:
    """Format a lookup result for display."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append(f"  Statute Lookup")
    lines.append("=" * 60)

    if not result["found"]:
        lines.append(f"\n  Act    : {result['act']}")
        lines.append(f"  Status : NOT FOUND")
        lines.append(f"  Note   : {result['text']}")
        lines.append(f"  Browse : {result['url']}")
    else:
        sec_label = f" s {result['section']}" if result['section'] else ""
        lines.append(f"\n  Act     : {result['act']}{sec_label}")
        lines.append(f"  Chapter : {result['cap']}")
        lines.append(f"  Source  : {result['url']}")
        lines.append(f"\n{result['text']}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def search_statutes(query: str) -> str:
    """
    Search AGC statutes online for a query term.
    Returns the raw search results page text for the LLM to parse.
    """
    search_url = f"{AGC_BASE}/Search/Current/All?SearchPhrase={quote(query)}"
    print(f"[STATUTE] Searching AGC for: {query}")
    html = _fetch_url(search_url)
    if not html:
        return "Search failed — check internet connection."
    return _strip_html(html)[:3000]


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        act     = sys.argv[1]
        section = sys.argv[2]
    elif len(sys.argv) == 2:
        act     = sys.argv[1]
        section = None
    else:
        # Default test
        act     = "MDA"
        section = "5"

    result = lookup_section(act, section)
    print(format_statute_result(result))