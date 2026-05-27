"""
legal_assistant.py — Standalone coding assistant for the Legal AI system.

This runs independently of app.py. It reads your project files from disk
and uses an LLM to review, troubleshoot, and propose improvements.

LLM options (in order of recommendation):
  1. Gemini 2.5 Flash via Google AI Studio (free, 1M context)
  2. Ollama llama3.1 (fully local, no internet required)

Setup for Gemini (recommended):
  pip install google-generativeai
  Get a free API key at https://aistudio.google.com/apikey
  Set environment variable: GEMINI_API_KEY=your_key_here

Setup for Ollama (local, no key needed):
  Uses your existing Ollama install.
"""
import os
import re
import json
import datetime

# ── LLM BACKEND SELECTION ─────────────────────────────────────────────────────

def _make_llm():
    """
    Return a callable: llm(prompt) -> str

    Tries Gemini first (best quality, free). Falls back to Ollama if no API key.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            def _gemini(prompt):
                resp = model.generate_content(prompt)
                return resp.text
            print("[Assistant] Using Gemini 2.5 Flash (Google AI Studio)")
            return _gemini
        except ImportError:
            print("[Assistant] google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            print(f"[Assistant] Gemini setup failed: {e}")

    # Fallback to local Ollama
    try:
        import requests
        def _ollama(prompt):
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.1", "prompt": prompt, "stream": False},
                timeout=120
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        print("[Assistant] Using Ollama llama3.1 (local)")
        return _ollama
    except Exception as e:
        print(f"[Assistant] Ollama unavailable: {e}")

    raise RuntimeError(
        "No LLM available. Set GEMINI_API_KEY environment variable "
        "or ensure Ollama is running at http://localhost:11434"
    )


# ── CONFIG ────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
IMPROVEMENT_LOG = os.path.join(PROJECT_ROOT, "data", "code_improvements.json")

SYSTEM_CONTEXT = """You are an expert Python developer assisting with a Singapore Legal AI system.

System architecture:
- LLM: llama3.1 via Ollama at http://localhost:11434 (langchain_ollama ChatOllama instance)
- Embeddings: nomic-embed-text via Ollama
- Vector store: custom JSON list at data/cases_db.json
- DB entry format: {"source": str, "folder": str, "text": str, "vector": list[float]}
- Pipeline: intent_router.py -> app.py -> legal_faiss.py -> legal_distiller.py -> reasoning_engine.py -> formatter.py

Coding standards:
- British spelling in all comments, docstrings, and print statements
- No markdown in LLM output — forbidden in prompts, stripped in post-processing
- Atomic file saves: write to .tmp then os.replace
- DB key is "text" (not "chunk") — this was a historical bug that has been fixed

When proposing changes:
- Always state which file is affected
- Always check cross-file impact before proposing
- Output complete files only — never partial snippets
- State paste order when multiple files change
"""

# ── IMPROVEMENT LOG ───────────────────────────────────────────────────────────

def _load_log() -> list:
    if not os.path.exists(IMPROVEMENT_LOG):
        return []
    try:
        with open(IMPROVEMENT_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_log(log: list):
    os.makedirs(os.path.dirname(IMPROVEMENT_LOG), exist_ok=True)
    tmp = IMPROVEMENT_LOG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    os.replace(tmp, IMPROVEMENT_LOG)

def _record(filename: str, description: str, applied: bool):
    log = _load_log()
    log.append({
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "file":        filename,
        "description": description,
        "applied":     applied,
    })
    _save_log(log)

# ── FILE HELPERS ──────────────────────────────────────────────────────────────

def _list_py_files() -> list:
    return sorted([
        f for f in os.listdir(PROJECT_ROOT)
        if f.endswith(".py") and os.path.isfile(os.path.join(PROJECT_ROOT, f))
    ])

def _read_file(filename: str) -> str:
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _write_file(filename: str, content: str):
    path = os.path.join(PROJECT_ROOT, filename)
    tmp  = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)

def _resolve_filename(query: str) -> str:
    available = _list_py_files()
    q = query.lower()
    for f in available:
        if f in q or f.replace(".py", "").replace("_", " ") in q:
            return f
    return ""

# ── HANDLERS ──────────────────────────────────────────────────────────────────

def _handle_review(query: str, llm) -> str:
    filename = _resolve_filename(query)
    if not filename:
        return (
            "Which file would you like me to review?\n"
            "Available: " + ", ".join(_list_py_files())
        )
    content = _read_file(filename)
    if not content:
        return f"Could not read '{filename}' from disk."

    print(f"[Assistant] Reading {filename} ({len(content):,} chars)...")

    prompt = (
        f"{SYSTEM_CONTEXT}\n\n"
        f"Please review the following file from the Legal AI project: {filename}\n\n"
        f"{content}\n\n"
        f"Provide a structured review covering:\n"
        f"1. Bugs or errors (with line numbers if possible)\n"
        f"2. Interface mismatches with other files in the system\n"
        f"3. Performance improvements\n"
        f"4. Code quality and maintainability\n"
        f"5. Specific improvements you recommend\n"
        f"\nDo not use markdown formatting in your response."
    )
    result = llm(prompt)
    _record(filename, "Code review", applied=False)
    return result

def _handle_rewrite(query: str, llm) -> str:
    filename = _resolve_filename(query)
    if not filename:
        return "Specify which file to rewrite, e.g.: rewrite ingest.py to use paragraph-level chunking"

    content = _read_file(filename)
    if not content:
        return f"Cannot read '{filename}' from disk."

    improvement = re.sub(re.escape(filename), "", query, flags=re.IGNORECASE).strip(" —-:")
    if not improvement:
        improvement = "general code quality, British spelling, and consistency improvements"

    print(f"[Assistant] Preparing rewrite of {filename}")
    print(f"[Assistant] Improvement: {improvement}")
    print(f"\n[Assistant] This will overwrite {filename} on disk.")
    confirm = input("  Proceed with rewrite? (y/n): ").strip().lower()
    if confirm not in ("y", "yes"):
        return "Rewrite cancelled."

    prompt = (
        f"{SYSTEM_CONTEXT}\n\n"
        f"Rewrite the following file to apply this improvement: {improvement}\n\n"
        f"File: {filename}\n\n"
        f"{content}\n\n"
        f"Output the complete rewritten file. Nothing else — no explanation before or after.\n"
        f"Do not wrap in markdown code fences."
    )

    print("[Assistant] Generating rewrite...")
    rewritten = llm(prompt)
    rewritten = re.sub(r"^```python\s*", "", rewritten, flags=re.MULTILINE)
    rewritten = re.sub(r"^```\s*$", "", rewritten, flags=re.MULTILINE)
    rewritten = rewritten.strip()

    _write_file(filename, rewritten)
    _record(filename, improvement, applied=True)

    return (
        f"{filename} rewritten and saved.\n"
        f"Improvement applied: {improvement}\n"
        f"Push to GitHub when satisfied:\n"
        f"  git add . && git commit -m \"{improvement}\" && git push origin master"
    )

def _handle_improve(query: str, llm) -> str:
    log = _load_log()
    past = "\n".join(
        f"- [{e['timestamp'][:10]}] {e['file']}: {e['description']}"
        for e in log[-10:]
    ) if log else "None yet."

    prompt = (
        f"{SYSTEM_CONTEXT}\n\n"
        f"Past improvements already applied:\n{past}\n\n"
        f"The user asked: {query}\n\n"
        f"Propose the 3 most impactful improvements not yet applied. "
        f"For each: state which file is affected, describe the change, "
        f"explain the expected benefit, and rate impact HIGH / MEDIUM / LOW.\n"
        f"Do not use markdown formatting."
    )
    print("[Assistant] Analysing system for improvement opportunities...")
    result = llm(prompt)
    _record("system", f"Self-improvement analysis: {query}", applied=False)
    return result

def _handle_troubleshoot(query: str, llm) -> str:
    # Read all core files to give the LLM full context
    core_files = [
        "legal_faiss.py", "legal_distiller.py", "intent_router.py",
        "reasoning_engine.py", "ingest.py", "app.py"
    ]
    context_parts = []
    for f in core_files:
        content = _read_file(f)
        if content:
            context_parts.append(f"=== {f} ===\n{content}")

    all_context = "\n\n".join(context_parts)

    prompt = (
        f"{SYSTEM_CONTEXT}\n\n"
        f"The user is reporting this problem:\n{query}\n\n"
        f"Here are the relevant project files:\n\n{all_context}\n\n"
        f"Diagnose the problem. Identify the root cause, which file contains it, "
        f"and provide the exact fix. "
        f"If multiple files need changing, state the paste order.\n"
        f"Do not use markdown formatting."
    )
    print("[Assistant] Reading core files and diagnosing...")
    return llm(prompt)

def _show_log() -> str:
    log = _load_log()
    if not log:
        return "No improvements recorded yet."
    lines = ["Improvement History", "=" * 40]
    for e in reversed(log[-20:]):
        status = "APPLIED" if e.get("applied") else "reviewed"
        lines.append(f"[{e['timestamp'][:16]}] [{status}] {e['file']}: {e['description']}")
    return "\n".join(lines)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

BANNER = """
============================================================
  Legal AI — Coding Assistant
  Commands:
    review <filename>     — Review a file for bugs and issues
    rewrite <filename>    — Rewrite a file with an improvement
    improve               — Propose system improvement strategies
    troubleshoot <error>  — Diagnose a bug or error
    list files            — Show all project files
    history               — Show improvement log
    exit                  — Quit
============================================================
"""

def main():
    llm = _make_llm()
    print(BANNER)

    while True:
        try:
            query = input("Assistant: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Exiting]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("[Session ended]")
            break

        q = query.lower()

        if q in ("list files", "show files"):
            print("\n".join(f"  {f}" for f in _list_py_files()))
        elif q in ("history", "improvement log"):
            print(_show_log())
        elif q.startswith("rewrite"):
            print(_handle_rewrite(query, llm))
        elif any(x in q for x in ("review", "check", "audit", "read")):
            print(_handle_review(query, llm))
        elif any(x in q for x in ("troubleshoot", "error", "problem", "not working", "broken", "failed")):
            print(_handle_troubleshoot(query, llm))
        else:
            print(_handle_improve(query, llm))


if __name__ == "__main__":
    main()