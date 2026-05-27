from llm import llm

# ── PROMPT TEMPLATES BY MODE ─────────────────────────────────────────────────

def _prompt_irac(query, context):
    return f"""You are a Singapore criminal law assistant.

Use ONLY the case law passages provided below. Do not cite any case or principle not found in the passages. Do not hallucinate.

QUESTION:
{query}

RETRIEVED CASE LAW:
{context}

Respond strictly in the following IRAC format:

Issue:
[Identify the precise legal issue raised by the question]

Rule:
[State the applicable legal rule or principle, citing the source case from the passages above]

Application:
[Apply the rule to the facts raised in the question, drawing directly from the passages]

Conclusion:
[State a clear conclusion]
"""

def _prompt_case_summary(query, context):
    return f"""You are a Singapore criminal law assistant.

Use ONLY the case law passages provided below. Do not add any facts or holdings not found in the passages.

REQUEST:
{query}

RETRIEVED PASSAGES:
{context}

Provide a structured case summary in this format:

Case Name:
[Full case name and citation if available in the passages]

Facts:
[Brief summary of the material facts]

Issue(s):
[Legal issues decided]

Holding:
[The court's decision]

Key Principle:
[The legal principle or ratio from this case]

Relevance:
[Why this case matters for Singapore criminal law]
"""

def _prompt_synthesis(query, context):
    return f"""You are a Singapore criminal law assistant.

Use ONLY the case law passages provided below. Do not cite any case not found in the passages.

REQUEST:
{query}

RETRIEVED PASSAGES:
{context}

Compare and synthesise the retrieved cases in this format:

Cases Considered:
[List each case found in the passages]

Points of Agreement:
[Where the cases are consistent with each other]

Points of Distinction:
[Where the cases differ, and why]

Synthesis:
[How these cases should be read together; which principle prevails and in what circumstances]

Conclusion:
[Practical takeaway for Singapore criminal practice]
"""

def _prompt_sentencing(query, context):
    return f"""You are a Singapore criminal law assistant specialising in sentencing.

Use ONLY the case law passages provided below. Do not invent sentencing figures or cases.

QUERY:
{query}

RETRIEVED SENTENCING PASSAGES:
{context}

Respond in this format:

Offence:
[Identify the offence being considered]

Sentencing Benchmark / Starting Point:
[State the benchmark or starting point from the cases, with citation]

Aggravating Factors (from cases):
[List aggravating factors identified in the retrieved passages]

Mitigating Factors (from cases):
[List mitigating factors identified in the retrieved passages]

Sentences Imposed:
[Summarise the actual sentences imposed across the retrieved cases]

Guidance:
[Practical sentencing guidance drawn from the passages]
"""

def _prompt_elements(query, context):
    return f"""You are a Singapore criminal law assistant.

Use ONLY the case law passages provided below.

QUERY:
{query}

RETRIEVED PASSAGES:
{context}

Set out the elements of the offence in this format:

Offence:
[Name and section of the offence]

Elements to be Proven by Prosecution:
[Number each element clearly]

Key Definitions:
[Define any terms of art from the passages]

Evidential Notes:
[Any notes on how the elements are typically proven, from the passages]

Case Authority:
[Which cases in the passages address these elements]
"""

def _prompt_procedure(query, context):
    return f"""You are a Singapore criminal law assistant.

Use ONLY the case law passages provided below.

QUERY:
{query}

RETRIEVED PASSAGES:
{context}

Answer the procedural question in this format:

Procedural Issue:
[Identify the procedural or evidential question]

Applicable Rule or Principle:
[State the rule from the passages]

Application:
[How the rule applies to the question asked]

Practical Guidance:
[Step-by-step practical guidance where applicable]

Authority:
[Cases from the passages supporting the above]
"""

def _prompt_drafting(query, context):
    return f"""You are a Singapore criminal law assistant helping draft legal submissions.

Use ONLY the case law passages provided below as your authorities. Do not cite cases not in the passages.

DRAFTING REQUEST:
{query}

RETRIEVED AUTHORITIES:
{context}

Produce the drafted content requested. Structure it professionally for use in Singapore criminal proceedings. Cite authorities from the passages inline where appropriate.

After the draft, include:

Authorities Used:
[List each case cited with a brief statement of what it stands for]
"""

def _prompt_definition(query, context):
    return f"""You are a Singapore criminal law assistant.

Use ONLY the case law passages provided below.

QUERY:
{query}

RETRIEVED PASSAGES:
{context}

Answer clearly and concisely in this format:

Definition / Explanation:
[Answer the question directly, in plain language]

Legal Basis:
[The rule, section, or principle from the passages]

Case Authority:
[Which cases in the passages support this definition]

Example (if available in passages):
[A concrete example from the cases illustrating the concept]
"""

# ── MODE DISPATCHER ──────────────────────────────────────────────────────────

PROMPT_MAP = {
    "irac":         _prompt_irac,
    "case_summary": _prompt_case_summary,
    "synthesis":    _prompt_synthesis,
    "sentencing":   _prompt_sentencing,
    "elements":     _prompt_elements,
    "procedure":    _prompt_procedure,
    "drafting":     _prompt_drafting,
    "definition":   _prompt_definition,
}

# ── PUBLIC API ────────────────────────────────────────────────────────────────

def reason(query: str, context_chunks: list, mode: str = "irac") -> str:
    """
    Run legal reasoning over retrieved chunks.

    Args:
        query:          The user's question
        context_chunks: List of chunk dicts from legal_distiller.distil()
                        Each dict must have a "chunk" key and "source" key
        mode:           Intent mode from intent_router.route()

    Returns:
        The LLM's structured response as a string
    """

    # Build context string from chunks
    lines = []
    for i, c in enumerate(context_chunks, 1):
        source = c.get("source", "Unknown")
        text   = c.get("chunk", c.get("text", ""))  # backward compatible
        lines.append(f"[{i}] {source}\n{text}")
    context = "\n\n".join(lines)

    if not context.strip():
        return "No relevant case law was found in the database for this query. Please ensure your PDF cases have been ingested."

    # Select prompt template
    prompt_fn = PROMPT_MAP.get(mode, _prompt_irac)
    prompt    = prompt_fn(query, context)

    return llm.invoke(prompt).content.strip()


if __name__ == "__main__":
    # Quick test
    test_chunks = [
        {
            "source": "test_case.pdf",
            "chunk": "The court held that for drug trafficking offences under s 5 MDA, the prosecution must prove that the accused had possession of the controlled drug, had knowledge of its nature, and that the possession was for the purpose of trafficking."
        }
    ]
    print(reason(
        query="What must the prosecution prove for drug trafficking?",
        context_chunks=test_chunks,
        mode="elements"
    ))