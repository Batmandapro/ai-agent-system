import os
import re
from llm import llm

# ── STYLE LEARNER INJECTION ───────────────────────────────────────────────────

def load_writing_rules():
    path = "data/profile/writing-rules.md"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

# ── CONTEXT BUILDER ───────────────────────────────────────────────────────────

def build_context(chunks):
    parts = []
    for c in chunks:
        source = c.get("source") or c.get("meta", {}).get("source", "Unknown")
        parts.append(f"[{source}]\n{c['text']}")
    return "\n\n".join(parts)

# ── SHARED INSTRUCTION ────────────────────────────────────────────────────────

SHARED = (
    "IMPORTANT: Do NOT use markdown formatting. "
    "Do NOT use **bold**, *italic*, or # headings. "
    "Use plain text only. "
    "Output section headings exactly as shown, ending with a colon, on their own line."
)

# ── PROMPT TEMPLATES ─────────────────────────────────────────────────────────

def prompt_irac(query, context, rules):
    style = f"\n\nWriting style instructions:\n{rules}" if rules else ""
    return f"""You are a Singapore criminal law assistant. Analyse the query using IRAC.
Use ONLY the provided context. Do NOT hallucinate cases or statutes.
Cite sources by name wherever possible.
{SHARED}{style}

QUERY:
{query}

CONTEXT:
{context}

Respond in this exact format:
Issue:
Rule:
Application:
Conclusion:
"""

def prompt_case_summary(query, context, rules):
    style = f"\n\nWriting style instructions:\n{rules}" if rules else ""
    return f"""You are a Singapore criminal law assistant. Provide a structured summary of the case referenced in the query.
Use ONLY the provided context. Do NOT hallucinate.
{SHARED}{style}

QUERY:
{query}

CONTEXT:
{context}

Respond in this exact format with each heading on its own line followed by the content:
Case Name:
Citation:
Court:
Facts:
Issue(s):
Holding:
Reasoning:
Relevance:
"""

def prompt_synthesis(query, context, rules):
    style = f"\n\nWriting style instructions:\n{rules}" if rules else ""
    return f"""You are a Singapore criminal law assistant. Compare and synthesise the cases or principles referenced.
Use ONLY the provided context. Do NOT hallucinate.
{SHARED}{style}

QUERY:
{query}

CONTEXT:
{context}

Respond in this exact format:
Cases Considered:
Points of Similarity:
Points of Distinction:
Reconciled Principle:
Conclusion:
"""

def prompt_sentencing(query, context, rules):
    style = f"\n\nWriting style instructions:\n{rules}" if rules else ""
    return f"""You are a Singapore criminal law assistant specialising in sentencing.
Identify the applicable sentencing framework, benchmarks and relevant precedents.
Use ONLY the provided context. Do NOT hallucinate.
{SHARED}{style}

QUERY:
{query}

CONTEXT:
{context}

Respond in this exact format:
Offence:
Statutory Range:
Sentencing Framework:
Relevant Precedents:
Aggravating Factors:
Mitigating Factors:
Indicative Sentence:
"""

def prompt_elements(query, context, rules):
    style = f"\n\nWriting style instructions:\n{rules}" if rules else ""
    return f"""You are a Singapore criminal law assistant. Set out the legal elements required to establish the offence or liability in question.
Use ONLY the provided context. Do NOT hallucinate.
{SHARED}{style}

QUERY:
{query}

CONTEXT:
{context}

Respond in this exact format:
Offence:
Statutory Provision:
Elements (Actus Reus):
Elements (Mens Rea):
Notes on Proof:
Key Cases:
"""

def prompt_procedure(query, context, rules):
    style = f"\n\nWriting style instructions:\n{rules}" if rules else ""
    return f"""You are a Singapore criminal law assistant. Explain the applicable criminal procedure clearly and accurately.
Use ONLY the provided context. Do NOT hallucinate.
{SHARED}{style}

QUERY:
{query}

CONTEXT:
{context}

Respond in this exact format:
Procedural Issue:
Applicable Provision(s):
Steps:
Relevant Timeline:
Notes:
"""

def prompt_drafting(query, context, rules):
    style = f"\n\nWriting style instructions:\n{rules}" if rules else ""
    return f"""You are a Singapore criminal law assistant. Draft the requested legal document or submission.
Use ONLY the provided context for legal propositions. Do NOT hallucinate authorities.
{SHARED}{style}

QUERY:
{query}

CONTEXT:
{context}

Draft:
"""

# ── DISPATCHER ────────────────────────────────────────────────────────────────

PROMPT_MAP = {
    "irac":         prompt_irac,
    "case_summary": prompt_case_summary,
    "synthesis":    prompt_synthesis,
    "sentencing":   prompt_sentencing,
    "elements":     prompt_elements,
    "procedure":    prompt_procedure,
    "drafting":     prompt_drafting,
}

def _strip_markdown(text):
    """Remove markdown bold, italic and heading markers from LLM output."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'^#+\s+',        '',    text, flags=re.MULTILINE)
    return text.strip()

def reason(query, context_chunks, mode="irac"):
    """Main entry point — selects prompt template by mode and calls LLM."""
    context   = build_context(context_chunks)
    rules     = load_writing_rules()
    prompt_fn = PROMPT_MAP.get(mode, prompt_irac)
    prompt    = prompt_fn(query, context, rules)
    raw       = llm.invoke(prompt).content
    return _strip_markdown(raw)