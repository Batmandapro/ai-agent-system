from llm import llm


def reason(query, context_chunks):
    context = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""
You are a Singapore law reasoning assistant.

Use ONLY the provided context.

Do NOT hallucinate authorities.

QUESTION:
{query}

CONTEXT:
{context}

FORMAT:
Issue:
Rule:
Application:
Conclusion:
"""

    return llm.invoke(prompt).content