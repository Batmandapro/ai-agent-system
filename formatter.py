import re
import datetime

# ── TABLE FORMATTER ───────────────────────────────────────────────────────────

def _wrap(text: str, width: int) -> list:
    """Word-wrap text to a given column width, returning a list of lines."""
    words  = text.split()
    lines  = []
    line   = ""
    for word in words:
        if len(line) + len(word) + 1 <= width:
            line = (line + " " + word).strip()
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]

def _table_row(label: str, content: str, col1: int = 12, col2: int = 60) -> str:
    """Render a two-column table row with word-wrapped content."""
    content_lines = _wrap(content, col2)
    rows = []
    for i, cline in enumerate(content_lines):
        if i == 0:
            left = f" {label:<{col1}}"
        else:
            left = " " * (col1 + 1)
        rows.append(f"│{left}│ {cline:<{col2}} │")
    return "\n".join(rows)

def format_case_summary_table(raw: str, sources: list, query: str) -> str:
    """
    Renders a case summary as a two-column table.

    Expected sections in raw: Case Name, Facts, Holding, Key Principle, Relevance, Remarks
    The case name occupies a full-width merged header row.
    """
    col1      = 12   # label column width
    col2      = 60   # content column width
    total     = col1 + col2 + 5  # borders + spaces
    divider   = "├" + "─" * (col1 + 2) + "┼" + "─" * (col2 + 2) + "┤"
    top       = "┌" + "─" * (total - 2) + "┐"
    bottom    = "└" + "─" * (total - 2) + "┘"

    # Extract case name from raw output
    case_name = _extract_section(raw, "Case Name") or query

    # Extract content sections
    sections = [
        ("Facts",     _extract_section(raw, "Facts")),
        ("Holding",   _extract_section(raw, "Holding")),
        ("Remarks",   _extract_section(raw, "Key Principle") or
                      _extract_section(raw, "Relevance") or
                      _extract_section(raw, "Remarks")),
    ]

    timestamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M")

    lines = []
    lines.append(f"\n  Case Summary  |  {timestamp}")
    lines.append(top)

    # Merged header — case name spans full width
    inner_width = total - 4
    name_lines  = _wrap(case_name, inner_width)
    for nl in name_lines:
        lines.append(f"│ {nl:<{inner_width}} │")

    for label, content in sections:
        if not content:
            continue
        lines.append(divider)
        lines.append(_table_row(label, content, col1, col2))

    lines.append(bottom)

    # Sources footer
    if sources:
        lines.append("\n  Sources:")
        for s in sources:
            lines.append(f"    • {s}")
    lines.append("")

    return "\n".join(lines)


def _extract_section(text: str, heading: str) -> str:
    """
    Extract the content under a given heading from the LLM output.
    Headings are assumed to end with a colon.
    """
    pattern = rf"{re.escape(heading)}:\s*(.*?)(?=\n[A-Z][^\n]*:|$)"
    match   = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


# ── GENERAL RESPONSE FORMATTER ────────────────────────────────────────────────

def format_response(raw: str, mode: str, sources: list, query: str) -> str:
    """
    Routes to table formatter for case_summary mode,
    or plain formatted output for all other modes.
    """

    if mode == "case_summary":
        return format_case_summary_table(raw, sources, query)

    output = []

    mode_labels = {
        "irac":       "IRAC Analysis",
        "synthesis":  "Case Synthesis",
        "sentencing": "Sentencing Analysis",
        "elements":   "Elements of Offence",
        "procedure":  "Procedural Guidance",
        "drafting":   "Drafted Submissions",
        "definition": "Legal Definition",
    }
    label     = mode_labels.get(mode, "Legal Analysis")
    timestamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M")

    output.append("=" * 60)
    output.append(f"  {label}")
    output.append(f"  {timestamp}")
    output.append("=" * 60)
    output.append(f"\nQuery: {query}\n")
    output.append("-" * 60)
    output.append(_clean(raw))

    if sources:
        output.append("\n" + "-" * 60)
        output.append("Sources relied upon:")
        for s in sources:
            output.append(f"  • {s}")

    output.append("=" * 60 + "\n")
    return "\n".join(output)


def format_error(message: str) -> str:
    return (
        "\n" + "=" * 60 + "\n"
        f"  [ERROR]\n  {message}\n"
        + "=" * 60 + "\n"
    )


def format_no_results(query: str) -> str:
    return (
        "\n" + "=" * 60 + "\n"
        f"  No relevant case law found\n"
        f"  Query: {query}\n\n"
        f"  Suggestions:\n"
        f"  • Try rephrasing your query\n"
        f"  • Check that ingest.py has been run\n"
        f"  • Add more cases to data/cases/\n"
        + "=" * 60 + "\n"
    )


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(
        r"(Issue:|Rule:|Application:|Conclusion:|Facts:|Holding:|"
        r"Remarks:|Case Name:|Synthesis:|Guidance:|Definition /.*?:)",
        r"\n\1", text
    )
    return text.strip()


if __name__ == "__main__":
    sample = """Case Name:
PP v Low Kok Heng [2007] 4 SLR(R) 183 (Court of Appeal)

Facts:
The accused was charged with drug trafficking. He was found in possession of heroin in a lorry. He claimed he did not know the nature of the drugs.

Holding:
The Court of Appeal held at [52] that the presumption under s 18(2) MDA operates such that once possession is proven, the accused is presumed to know the nature of the drug. The burden shifts to the accused to rebut on a balance of probabilities.

Key Principle:
The s 18(2) MDA presumption is a critical prosecutorial tool. The accused must adduce credible evidence of lack of knowledge to rebut it."""

    print(format_case_summary_table(
        raw=sample,
        sources=["Low Kok Heng [2007] 4 SLR(R) 0183.pdf"],
        query="Summarise the case of Low Kok Heng"
    ))