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

# ── SHARED ANTI-MARKDOWN INSTRUCTION ─────────────────────────────────────────

SHARED = (
    "CRITICAL FORMATTING RULES — you must follow these exactly:\n"
    "1. Do NOT use any markdown formatting whatsoever.\n"
    "2. Do NOT use **bold** (double asterisks) under any circumstances.\n"
    "3. Do NOT use *italic* (single asterisks) under any circumstances.\n"
    "4. Do NOT use # headings (hash symbols) under any circumstances.\n"
    "5. Do NOT use bullet points with - or * symbols.\n"
    "6. Use plain text only. Section headings must appear exactly as shown "
    "in the format below, ending with a colon, on their own line.\n"
    "7. Write in British English spelling throughout "
    "(e.g. 'analyse' not 'analyze', 'favour' not 'favor', "
    "'recognised' not 'recognized').\n"
)

# ── PROMPT TEMPLATES ──────────────────────────────────────────────────────────

def prompt_irac(query, context, rules):
    style = f"\n\nWriting style instructions:\n{rules}" if rules else ""
    return f"""You are a Singapore law assistant. Analyse the query using IRAC.
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
    return f"""You are a Singapore law assistant. Provide a structured summary of the case referenced in the query.
Use ONLY the provided context. Do NOT hallucinate details not present in the context.
{SHARED}{style}

PARAGRAPH CITATION REQUIREMENT:
You must identify and cite paragraph numbers from the judgment text wherever they appear in the context.
Paragraph numbers appear in square brackets, for example [14], [23], [52].
When stating a holding, ratio, or key finding, cite the paragraph number as follows:
"The court held at [52] that..." or "As noted at [14], the principle is..."
If no paragraph numbers are visible in the context, state "paragraph numbers not available in retrieved text".

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
    return f"""You are a Singapore law assistant. Compare and synthesise the cases or principles referenced.
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
    return f"""You are a Singapore law assistant specialising in sentencing.
Identify the applicable sentencing framework, benchmarks and relevant precedents.
Use ONLY the provided context. Do NOT hallucinate.
{SHARED}{style}

PARAGRAPH CITATION REQUIREMENT:
When citing a holding, framework, or ratio, cite the paragraph number if visible:
"The court held at [14] that the starting point is..."

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
    return f"""You are a Singapore law assistant. Set out the legal elements required to establish the offence or liability in question.
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
    return f"""You are a Singapore law assistant. Explain the applicable procedure clearly and accurately.
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
    return f"""You are a Singapore law assistant. Draft the requested legal document or submission.
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
    """Remove markdown formatting from LLM output as a post-processing safety net."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*\s+', '  ', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*', '', text)
    return text.strip()

def reason(query, context_chunks, mode="irac"):
    """Main entry point — selects prompt template by mode and calls LLM."""
    context   = build_context(context_chunks)
    rules     = load_writing_rules()
    prompt_fn = PROMPT_MAP.get(mode, prompt_irac)
    prompt    = prompt_fn(query, context, rules)
    raw       = llm.invoke(prompt).content
    return _strip_markdown(raw)