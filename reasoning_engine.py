# FILE: reasoning_engine.py
# REPLACES: C:\Users\Admin\Desktop\ai-agent-system\reasoning_engine.py

"""
reasoning_engine.py

Builds intent-specific prompts for the Singapore Legal AI system and calls
the LLM (llama3.1 via Ollama). Returns plain text — no markdown, no bullet
points, no headers in LLM output. This is a hard constraint enforced by
every prompt template.

Improvements in this version:
  1. Loads data/legal_principles.md at module level and injects it into
     every prompt as authoritative background knowledge.
  2. Singapore-specific IRAC prompt with precise framing instructions.
  3. Anti-hallucination instruction added to every prompt.
  4. Per-intent prompt templates for all seven intents.
  5. Citation format instruction: "In CaseName [Year] COURT No (at [para_ref]),
     the court held that..."
  6. Paragraph references from chunks are surfaced and cited correctly.
"""

import os
import requests

# ---------------------------------------------------------------------------
# Module-level load of legal principles
# ---------------------------------------------------------------------------

_LEGAL_PRINCIPLES: str = ""

_PRINCIPLES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "legal_principles.md"
)

try:
    with open(_PRINCIPLES_PATH, "r", encoding="utf-8") as _f:
        _LEGAL_PRINCIPLES = _f.read().strip()
except FileNotFoundError:
    pass  # Graceful fallback — principles file not yet present
except Exception as _e:
    print(f"[reasoning_engine] Warning: could not load legal principles — {_e}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1"
TIMEOUT = 120

ANTI_HALLUCINATION = (
    "You must only cite cases that appear in the RETRIEVED CASES section below. "
    "Do not invent case names, citations, or paragraph numbers. "
    "If you cannot answer from the provided cases, state: "
    "Insufficient case authority provided — please ingest additional cases on this topic."
)

CITATION_FORMAT = (
    "When citing a case, always write it in this form: "
    "'In CaseName [Year] COURT No (at [para_ref]), the court held that...'. "
    "If no paragraph reference is available, omit the '(at [para_ref])' part entirely."
)

OUTPUT_FORMAT = (
    "Your response must be plain prose only. "
    "Do not use markdown, bullet points, numbered lists, bold, italics, or any headers. "
    "Write in continuous paragraphs as you would in a formal legal memorandum."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_principles_block() -> str:
    """Return the legal principles block for injection into prompts."""
    if not _LEGAL_PRINCIPLES:
        return ""
    return (
        "ESTABLISHED PRINCIPLES — TREAT AS AUTHORITATIVE\n\n"
        + _LEGAL_PRINCIPLES
        + "\n\n"
    )


def _build_cases_block(chunks: list) -> str:
    """
    Convert distilled chunks into a plain-text cases block.

    Each chunk is a dict with keys: case_name, citation, para_ref, text,
    score, source, folder. para_ref may be None or an empty string.
    """
    if not chunks:
        return "RETRIEVED CASES\n\nNo cases have been retrieved for this query.\n\n"

    lines = ["RETRIEVED CASES\n"]
    for i, chunk in enumerate(chunks, start=1):
        case_name = chunk.get("case_name", "Unknown")
        citation = chunk.get("citation", "")
        para_ref = chunk.get("para_ref", "")
        text = chunk.get("text", "")

        header = f"[{i}] {case_name}"
        if citation:
            header += f" {citation}"
        if para_ref:
            header += f" at {para_ref}"

        lines.append(header)
        lines.append(text.strip())
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Per-intent prompt builders
# ---------------------------------------------------------------------------

def _prompt_irac(query: str, cases_block: str, principles_block: str) -> str:
    return f"""{principles_block}{ANTI_HALLUCINATION}

{CITATION_FORMAT}

{OUTPUT_FORMAT}

You are a senior Singapore lawyer preparing a legal memorandum under Singapore law. \
The client has asked the following question:

QUERY: {query}

Using only the retrieved cases below, produce a rigorous IRAC analysis structured as follows.

Issue: Identify the precise legal issue as it would be framed before a Singapore court. \
State the issue in one or two sentences, identifying the relevant statute, common law rule, \
or equitable doctrine under Singapore law. Do not frame the issue under English, Australian, \
or any foreign law unless that foreign law has been expressly adopted by Singapore courts.

Rule: State the applicable rule as it stands in Singapore law. \
Identify the leading Singapore authority and, where relevant, note whether the rule \
originates from statute or common law. \
If Singapore courts have modified or departed from English positions, note that departure.

Application: Apply the rule to the facts raised in the query using only the retrieved cases. \
Cite every case you rely on in the form required by the citation format instruction above. \
Where a paragraph reference is available, you must include it. \
Reason through the facts methodically.

Conclusion: State the likely outcome, the degree of certainty, \
and any material uncertainties or gaps in the case authority provided.

{cases_block}"""


def _prompt_sentencing(query: str, cases_block: str, principles_block: str) -> str:
    return f"""{principles_block}{ANTI_HALLUCINATION}

{CITATION_FORMAT}

{OUTPUT_FORMAT}

You are a senior Singapore criminal lawyer advising on sentencing. \
The query is:

QUERY: {query}

Using only the retrieved cases below, provide a sentencing analysis covering: \
the applicable sentencing range for this offence under Singapore law; \
the benchmark case or cases that establish the starting point, cited with paragraph references; \
the aggravating factors identified in the retrieved cases and their typical impact on sentence; \
the mitigating factors identified in the retrieved cases and their typical impact on sentence; \
and a reasoned estimate of the likely sentencing outcome on the facts raised in the query. \
If the retrieved cases do not provide sufficient guidance to cover any of these points, \
state that explicitly rather than speculating.

{cases_block}"""


def _prompt_elements(query: str, cases_block: str, principles_block: str) -> str:
    return f"""{principles_block}{ANTI_HALLUCINATION}

{CITATION_FORMAT}

{OUTPUT_FORMAT}

You are a senior Singapore lawyer. The query is:

QUERY: {query}

Using only the retrieved cases below, identify and explain every legal element \
that the prosecution or claimant must prove to establish the cause of action or offence \
raised in the query. For each element, state: \
what the element requires; \
which party bears the burden of proof and to what standard; \
and the case authority that established or confirmed that element under Singapore law, \
cited with its paragraph reference where available. \
If an element has not been addressed in the retrieved cases, say so explicitly.

{cases_block}"""


def _prompt_summary(query: str, cases_block: str, principles_block: str) -> str:
    return f"""{principles_block}{ANTI_HALLUCINATION}

{CITATION_FORMAT}

{OUTPUT_FORMAT}

You are a senior Singapore lawyer. The query is:

QUERY: {query}

Using only the retrieved cases below, write a concise three-paragraph summary of \
what those cases collectively establish under Singapore law. \
The first paragraph should describe the core legal rule or principle they establish. \
The second paragraph should describe how courts have applied that rule in practice, \
noting the most significant factual distinctions between the cases. \
The third paragraph should identify any residual uncertainties or open questions \
that the cases leave unresolved. \
Cite cases in the form required by the citation format instruction above.

{cases_block}"""


def _prompt_synthesis(query: str, cases_block: str, principles_block: str) -> str:
    return f"""{principles_block}{ANTI_HALLUCINATION}

{CITATION_FORMAT}

{OUTPUT_FORMAT}

You are a senior Singapore lawyer. The query is:

QUERY: {query}

Using only the retrieved cases below, produce a synthesis of the case law. \
First, identify the points of agreement across the cases — the propositions that every \
retrieved case accepts without qualification. \
Second, identify the points of tension or apparent conflict — cases that reach \
different conclusions, apply different tests, or interpret the same rule differently. \
Third, where there is tension, propose the most coherent way to reconcile the cases, \
or state clearly that the law in this area is unsettled. \
Cite every case you rely on in the form required by the citation format instruction, \
including paragraph references where available.

{cases_block}"""


def _prompt_procedure(query: str, cases_block: str, principles_block: str) -> str:
    return f"""{principles_block}{ANTI_HALLUCINATION}

{CITATION_FORMAT}

{OUTPUT_FORMAT}

You are a senior Singapore litigation lawyer. The query is:

QUERY: {query}

Using only the retrieved cases below, explain the procedural steps, timelines, \
and filing requirements that are relevant to the query. \
Where the retrieved cases discuss procedural requirements, cite them with paragraph \
references in the form required by the citation format instruction. \
Where a procedural step is governed by statute or the Rules of Court, identify \
the specific provision. \
If the retrieved cases do not address a procedural point raised in the query, say so explicitly \
rather than speculating from general knowledge.

{cases_block}"""


def _prompt_drafting(query: str, cases_block: str, principles_block: str) -> str:
    return f"""{principles_block}{ANTI_HALLUCINATION}

{CITATION_FORMAT}

{OUTPUT_FORMAT}

You are a senior Singapore lawyer drafting a legal document. The instruction is:

QUERY: {query}

Using the legal principles established in the retrieved cases below, \
produce a plain-text draft of the clause, letter, or document requested. \
After the draft, include a short explanatory note — written as continuous prose, \
not as a list — identifying the legal principles from the retrieved cases \
that the draft is based on, with citations in the form required by the citation format \
instruction above. \
If the retrieved cases provide insufficient authority to support a particular \
provision in the draft, note that in the explanatory section.

{cases_block}"""


# ---------------------------------------------------------------------------
# Prompt dispatcher
# ---------------------------------------------------------------------------

_PROMPT_BUILDERS = {
    "IRAC": _prompt_irac,
    "SENTENCING": _prompt_sentencing,
    "ELEMENTS": _prompt_elements,
    "SUMMARY": _prompt_summary,
    "SYNTHESIS": _prompt_synthesis,
    "PROCEDURE": _prompt_procedure,
    "DRAFTING": _prompt_drafting,
}

_DEFAULT_INTENT = "IRAC"


def _build_prompt(query: str, intent: str, chunks: list) -> str:
    """Select the correct prompt builder and assemble the full prompt."""
    principles_block = _build_principles_block()
    cases_block = _build_cases_block(chunks)

    builder = _PROMPT_BUILDERS.get(intent.upper(), _PROMPT_BUILDERS[_DEFAULT_INTENT])
    return builder(query, cases_block, principles_block)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """
    Send the prompt to Ollama (llama3.1) and return the response text.
    Raises RuntimeError if the call fails.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "LLM request timed out after 120 seconds. "
            "The model may be overloaded — please try again."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama at http://localhost:11434. "
            "Ensure Ollama is running and the model is loaded."
        )
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def reason(query: str, intent: str, chunks: list) -> str:
    """
    Build an intent-specific prompt and call the LLM.

    Parameters
    ----------
    query   : The user's legal question.
    intent  : One of IRAC, SENTENCING, ELEMENTS, SUMMARY, SYNTHESIS,
              PROCEDURE, DRAFTING. Unknown intents fall back to IRAC.
    chunks  : List of dicts from the retrieval pipeline. Each dict must
              contain at minimum: case_name, citation, para_ref, text.
              Additional keys (score, source, folder) are accepted but
              not required for prompt construction.

    Returns
    -------
    Plain text response from the LLM. No markdown, no bullet points,
    no headers. The formatter.py module receives this string directly.
    """
    intent_upper = intent.upper() if intent else _DEFAULT_INTENT
    if intent_upper not in _PROMPT_BUILDERS:
        print(
            f"[reasoning_engine] Unknown intent '{intent}' — "
            f"falling back to {_DEFAULT_INTENT}."
        )
        intent_upper = _DEFAULT_INTENT

    prompt = _build_prompt(query, intent_upper, chunks)

    print(
        f"[reasoning_engine] Calling LLM for intent={intent_upper}, "
        f"chunks={len(chunks)}, principles_loaded={bool(_LEGAL_PRINCIPLES)}"
    )

    return _call_llm(prompt)
