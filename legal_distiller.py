# FILE: legal_distiller.py
# REPLACES: C:\Users\Admin\Desktop\ai-agent-system\legal_distiller.py

import re


# ---------------------------------------------------------------------------
# MODULE DOCSTRING
# ---------------------------------------------------------------------------
"""
legal_distiller.py

Distils raw text chunks retrieved from the vector database into structured
dicts suitable for consumption by reasoning_engine.py.

Each incoming chunk is expected to be a dict with the keys:
    source    : str   -- filename of the source document
    folder    : str   -- folder / collection the document belongs to
    text      : str   -- raw text of the retrieved chunk
    vector    : list  -- embedding vector (not used here)
    score     : float -- cosine or dot-product similarity score
    para_start: int | None -- first paragraph number set during ingestion
                              (absent in older DB entries; handled gracefully)

Each outgoing distilled result is a dict with the keys:
    case_name : str
    citation  : str
    para_ref  : str   -- e.g. "[14]", "[14]-[17]", or ""
    text      : str   -- cleaned chunk text (paragraph markers preserved)
    score     : float
    source    : str
    folder    : str

British spelling is used throughout all comments and docstrings.
No LLM is invoked; this module is pure text processing.
"""


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Singapore court abbreviations recognised in neutral citations.
_SG_COURTS = r"SGCA|SGHC|SGDC|SGMC|SGHCF|SGHCR"

# Full neutral-citation pattern, optionally followed by "at [N]".
# Examples:
#   PP v Tan Kiam Peng [2007] SGCA 38
#   Spandeck Engineering (S) Pte Ltd v Defence Science & Technology Agency
#       [2007] SGCA 37 at [15]
_CITATION_RE = re.compile(
    r"\[(\d{4})\]\s+(?:" + _SG_COURTS + r")\s+\d+"
    r"(?:\s+at\s+\[\d+\])?",
    re.IGNORECASE,
)

# Paragraph-number patterns used in Singapore judgments.
# Style A: [45] at the start of a line (modern neutral citation style).
# Style B: 45. or 45  at the start of a line (older judgments).
_PARA_BRACKET_RE = re.compile(r"^\[(\d+)\]", re.MULTILINE)
_PARA_PLAIN_RE = re.compile(r"^(\d+)[.\s]", re.MULTILINE)

# Lines that are purely page headers / footers to be stripped.
# Matches: bare page numbers, "Page 3 of 47", "3 of 47".
_PAGE_LINE_RE = re.compile(
    r"^\s*(?:Page\s+)?\d+\s*(?:of\s+\d+)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Three or more consecutive blank lines collapsed to two.
_EXCESS_BLANK_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# PRIVATE HELPERS
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Clean a raw chunk text by:
    - Removing page-header / page-footer lines (bare numbers, "Page X of Y").
    - Collapsing runs of more than two consecutive blank lines to two.
    - Stripping leading and trailing whitespace from the whole block.

    Paragraph markers such as [1] or [2] embedded within a line are
    deliberately preserved so that reasoning_engine.py can cite them.
    """
    # Remove lines that are solely page numbers or "Page X of Y".
    text = _PAGE_LINE_RE.sub("", text)

    # Collapse excessive blank lines.
    text = _EXCESS_BLANK_RE.sub("\n\n", text)

    return text.strip()


def _extract_para_numbers(chunk: dict) -> str:
    """
    Determine the paragraph reference string for a chunk.

    Strategy (in order of priority):
    1. Use chunk["para_start"] if it is present and not None.
       This value is set by the ingest pipeline at document ingestion time
       and is the most reliable source.
    2. Fall back to scanning chunk["text"] for bracketed paragraph markers
       of the form [N] at the start of a line.
    3. If no bracketed markers are found, scan for plain numeric markers
       of the form "N." or "N " at the start of a line.
    4. If nothing is found, return an empty string.

    When a single paragraph number is identified, the result is "[N]".
    When a range is identified (multiple numbers in one chunk), the result
    is "[N]-[M]" where N is the lowest and M is the highest number found.
    """
    # --- Priority 1: para_start from ingestion metadata ---
    para_start = chunk.get("para_start", None)
    if para_start is not None:
        # para_start gives us the first paragraph in the chunk.
        # Scan the text for any additional paragraph markers to build a range.
        text = chunk.get("text", "")
        bracket_hits = [int(m) for m in _PARA_BRACKET_RE.findall(text)]

        if bracket_hits:
            # Combine the ingestion-supplied start with whatever we find in text.
            all_nums = sorted(set([para_start] + bracket_hits))
        else:
            all_nums = [para_start]

        if len(all_nums) == 1:
            return "[{}]".format(all_nums[0])
        else:
            return "[{}]-[{}]".format(all_nums[0], all_nums[-1])

    # --- Priority 2: bracketed markers [N] in text ---
    text = chunk.get("text", "")
    bracket_hits = [int(m) for m in _PARA_BRACKET_RE.findall(text)]
    if bracket_hits:
        bracket_hits = sorted(set(bracket_hits))
        if len(bracket_hits) == 1:
            return "[{}]".format(bracket_hits[0])
        return "[{}]-[{}]".format(bracket_hits[0], bracket_hits[-1])

    # --- Priority 3: plain numeric markers N. or N at line start ---
    plain_hits = [int(m) for m in _PARA_PLAIN_RE.findall(text)]
    if plain_hits:
        plain_hits = sorted(set(plain_hits))
        # Sanity-check: ignore if numbers look like years or page numbers
        # (i.e., four-digit numbers >= 1900).
        plain_hits = [n for n in plain_hits if not (1900 <= n <= 2100)]
        if plain_hits:
            if len(plain_hits) == 1:
                return "[{}]".format(plain_hits[0])
            return "[{}]-[{}]".format(plain_hits[0], plain_hits[-1])

    return ""


def extract_case_name(text: str) -> str:
    """
    Attempt to extract the case name from the supplied text.

    Singapore judgments typically place the case name on the first
    non-blank line, before the neutral citation.  The function:
    1. Searches for a line immediately preceding a recognised citation.
    2. Falls back to returning the first non-blank line of the text.

    Returns an empty string if no candidate is found.

    This function is part of the public API called by api.py and app.py.
    """
    if not text:
        return ""

    lines = text.splitlines()

    # Try to find a line immediately before a citation marker.
    for i, line in enumerate(lines):
        if _CITATION_RE.search(line):
            # The case name is usually on this same line before the citation,
            # or on the immediately preceding non-blank line.
            # First check same line.
            candidate = _CITATION_RE.sub("", line).strip().rstrip(",;:")
            if candidate:
                return candidate
            # Check the line above.
            for j in range(i - 1, -1, -1):
                prev = lines[j].strip()
                if prev:
                    return prev
            break

    # Fall back: first non-blank line.
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped

    return ""


def _extract_citation(text: str) -> str:
    """
    Extract the first Singapore neutral citation found in the text.

    Handles:
    - Standard form:  [Year] COURT Number
    - With paragraph: [Year] COURT Number at [N]

    Returns an empty string if no citation is found.
    """
    match = _CITATION_RE.search(text)
    if match:
        return match.group(0).strip()
    return ""


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def distil(chunks: list) -> list:
    """
    Distil a list of raw retrieval chunks into structured result dicts.

    Parameters
    ----------
    chunks : list of dict
        Each dict must contain at minimum the keys "source", "folder",
        "text", and "score".  The "para_start" key is optional and may
        be absent in chunks retrieved from an older index that was built
        without the updated ingest.py.

    Returns
    -------
    list of dict
        Each dict contains:
            case_name : str
            citation  : str
            para_ref  : str
            text      : str  (cleaned, paragraph markers preserved)
            score     : float
            source    : str
            folder    : str
    """
    results = []

    for chunk in chunks:
        raw_text = chunk.get("text", "")
        cleaned = _clean_text(raw_text)

        case_name = extract_case_name(cleaned)
        citation = _extract_citation(cleaned)
        para_ref = _extract_para_numbers(chunk)

        results.append({
            "case_name": case_name,
            "citation": citation,
            "para_ref": para_ref,
            "text": cleaned,
            "score": chunk.get("score", 0.0),
            "source": chunk.get("source", ""),
            "folder": chunk.get("folder", ""),
        })

    return results
