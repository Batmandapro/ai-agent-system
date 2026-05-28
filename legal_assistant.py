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


# ---------------------------------------------------------------------------
# Entity scanner — reads project files at startup to extract live values
# ---------------------------------------------------------------------------

def _scan_project_entities(project_root):
    """Scan all .py files and extract key entities.

    Returns a dict of discovered values: LLM models, URLs, DB paths,
    embedding models, class names, and top-level function names.
    Injected into every prompt so the LLM never guesses.
    """
    entities = {
        "llm_models":       [],
        "embedding_models": [],
        "ollama_urls":      [],
        "db_paths":         [],
        "db_keys":          [],
        "flask_port":       None,
        "classes":          {},
        "functions":        {},
        "file_list":        [],
    }

    py_files = sorted(
        f for f in os.listdir(project_root)
        if f.endswith(".py") and os.path.isfile(os.path.join(project_root, f))
    )
    entities["file_list"] = py_files

    for fname in py_files:
        path = os.path.join(project_root, fname)
        try:
            src = open(path, "r", encoding="utf-8").read()
        except OSError:
            continue

        for m in re.findall(r'["\']([a-zA-Z0-9][a-zA-Z0-9.:\-_/]+)["\']', src):
            if any(kw in m.lower() for kw in ("llama", "gemini", "gpt", "claude", "mistral", "phi", "nomic", "embed")):
                target = entities["llm_models"] if "embed" not in m.lower() else entities["embedding_models"]
                if m not in target:
                    target.append(m)

        for u in re.findall(r'["\']https?://[^"\' ]+["\']', src):
            u = u.strip("'\"")
            if u not in entities["ollama_urls"]:
                entities["ollama_urls"].append(u)

        for p in re.findall(r'["\']data/[^"\' ]+\.json["\']', src):
            p = p.strip("'\"")
            if p not in entities["db_paths"]:
                entities["db_paths"].append(p)

        for k in re.findall(r'\[[\'"](text|chunk|source|folder|vector)[\'"]\]', src):
            if k not in entities["db_keys"]:
                entities["db_keys"].append(k)

        port_match = re.search(r'port\s*=\s*(\d+)', src)
        if port_match and entities["flask_port"] is None:
            entities["flask_port"] = port_match.group(1)

        classes = re.findall(r'^class\s+(\w+)', src, re.MULTILINE)
        if classes:
            entities["classes"][fname] = classes

        funcs = re.findall(r'^def\s+(\w+)', src, re.MULTILINE)
        if funcs:
            entities["functions"][fname] = funcs

    return entities


def _build_live_context(project_root):
    """Return a formatted plain-text block of live project entities.

    Prepended to every prompt so the LLM works from actual values.
    """
    e = _scan_project_entities(project_root)
    lines = [
        "LIVE PROJECT ENTITIES (scanned from disk — use these exact values, never placeholders):",
        "",
        f"Project files present: {', '.join(e['file_list']) if e['file_list'] else '(none found)'}",
    ]
    if e["llm_models"]:
        lines.append(f"LLM models in use: {', '.join(e['llm_models'])}")
    if e["embedding_models"]:
        lines.append(f"Embedding models in use: {', '.join(e['embedding_models'])}")
    if e["ollama_urls"]:
        lines.append(f"API / Ollama URLs in use: {', '.join(e['ollama_urls'])}")
    if e["db_paths"]:
        lines.append(f"Database file paths: {', '.join(e['db_paths'])}")
    if e["db_keys"]:
        lines.append(f"Database dict keys confirmed in source: {', '.join(e['db_keys'])}")
    if e["flask_port"]:
        lines.append(f"Flask server port: {e['flask_port']}")
    if e["classes"]:
        lines.append("")
        lines.append("Classes per file:")
        for fname, cls_list in e["classes"].items():
            lines.append(f"  {fname}: {', '.join(cls_list)}")
    if e["functions"]:
        lines.append("")
        lines.append("Top-level functions per file:")
        for fname, fn_list in e["functions"].items():
            lines.append(f"  {fname}: {', '.join(fn_list)}")
    return "\n".join(lines)

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

DO NOT RUSH. Think carefully and thoroughly before responding. Consider the full consequences
of every suggestion before writing a single line of output.

System architecture:
- LLM: llama3.1 via Ollama at http://localhost:11434 (langchain_ollama ChatOllama instance)
- Embeddings: nomic-embed-text via Ollama
- Vector store: custom JSON list at data/cases_db.json
- DB entry format: {"source": str, "folder": str, "text": str, "vector": list[float]}
- Pipeline: intent_router.py -> app.py -> legal_faiss.py -> legal_distiller.py -> reasoning_engine.py -> formatter.py
- REST API: api.py (Flask, port 5000)
- Standalone coding assistant: legal_assistant.py

Coding standards:
- British spelling in all comments, docstrings, and print statements
- No markdown in LLM output — forbidden in prompts, stripped in post-processing
- Atomic file saves: write to .tmp then os.replace
- DB key is "text" (not "chunk") — this was a historical bug that has been fixed
- All optional imports wrapped in try/except with a boolean flag

CRITICAL RULE — IMPACT ASSESSMENT REQUIRED:
Before making ANY suggestion, fix, or rewrite, you MUST first assess the full pipeline impact.
For every proposed change, explicitly state:
1. IMPACT ASSESSMENT: Which other files in the pipeline are affected by this change (if any), and why.
2. REQUIRED ALIGNED CHANGES: The exact corresponding changes needed in those other files to keep the system consistent.
3. Only then present the suggested fix or rewrite.
If a change affects no other files, you must still explicitly state "No other files are affected" so the developer knows the assessment was done.

CRITICAL RULE — NO PLACEHOLDERS:
NEVER use placeholder values such as "your_model_here", "<insert_key>", "TODO", "...", or any
stand-in that leaves the code non-functional. Every value you write must be the actual, correct
value already used by this project, as confirmed in the LIVE PROJECT ENTITIES section injected
into each prompt. If you are unsure of a value, say so explicitly in plain text BEFORE the code.

CRITICAL RULE — CODE DELIMITERS:
Whenever you output code, you MUST wrap it exactly as follows:

=== CODE BEGIN: <filename> ===
<code here>
=== CODE END: <filename> ===

This applies to every piece of code without exception.

Output rules:
- Always output complete files only — never partial snippets
- State paste order when multiple files change
- Do not use markdown formatting in responses
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
    print("[Assistant] Scanning project entities...")
    live_ctx = _build_live_context(PROJECT_ROOT)

    prompt = (
        f"{SYSTEM_CONTEXT}\n\n"
        f"{live_ctx}\n\n"
        f"Please review the following file from the Legal AI project: {filename}\n\n"
        f"Provide a structured review covering:\n"
        f"1. Bugs or errors (with line numbers if possible)\n"
        f"2. Interface mismatches with other files in the system\n"
        f"3. Performance improvements\n"
        f"4. Code quality and maintainability\n"
        f"5. Specific improvements you recommend\n\n"
        f"For each issue or recommendation, you MUST state:\n"
        f"  IMPACT ASSESSMENT: Which other pipeline files are affected by this issue or change.\n"
        f"  REQUIRED ALIGNED CHANGES: Exactly what must change in those files to keep the system consistent.\n"
        f"  If no other files are affected, explicitly state: No other files are affected.\n\n"
        f"Wrap any code suggestions in: === CODE BEGIN: <filename> === ... === CODE END: <filename> ===\n"
        f"Do not use markdown formatting in your response.\n\n"
        f"=== FILE: {filename} ===\n{content}\n=== END FILE ==="
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

    print("[Assistant] Scanning project entities...")
    live_ctx = _build_live_context(PROJECT_ROOT)

    prompt = (
        f"{SYSTEM_CONTEXT}\n\n"
        f"{live_ctx}\n\n"
        f"Rewrite the following file to apply this improvement: {improvement}\n\n"
        f"BEFORE providing the rewritten code, produce a plain-text IMPACT ASSESSMENT:\n"
        f"  - List every other file in the pipeline that will need a corresponding change.\n"
        f"  - State exactly what must change in each of those files.\n"
        f"  - If no other files are affected, explicitly state: No other files are affected.\n\n"
        f"Then output the complete rewritten file wrapped in delimiters:\n"
        f"=== CODE BEGIN: {filename} ===\n<code>\n=== CODE END: {filename} ===\n\n"
        f"=== FILE: {filename} ===\n{content}\n=== END FILE ==="
    )

    print("[Assistant] Generating rewrite...")
    rewritten_raw = llm(prompt)

    # Extract only the code between the delimiters to write to disk
    code_match = re.search(
        rf"=== CODE BEGIN: {re.escape(filename)} ===\n(.*?)\n=== CODE END: {re.escape(filename)} ===",
        rewritten_raw, re.DOTALL
    )
    if code_match:
        rewritten = code_match.group(1).strip()
    else:
        # Fallback: strip markdown fences if delimiters weren't used
        rewritten = re.sub(r"^```python\s*", "", rewritten_raw, flags=re.MULTILINE)
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

    print("[Assistant] Scanning project entities...")
    live_ctx = _build_live_context(PROJECT_ROOT)

    prompt = (
        f"{SYSTEM_CONTEXT}\n\n"
        f"{live_ctx}\n\n"
        f"Past improvements already applied:\n{past}\n\n"
        f"The user asked: {query}\n\n"
        f"Propose the 3 most impactful improvements not yet applied.\n"
        f"For each improvement:\n"
        f"  - State which file is affected\n"
        f"  - Describe the change\n"
        f"  - Provide an IMPACT ASSESSMENT: which other pipeline files are affected and what must change in them\n"
        f"  - If no other files are affected, explicitly state: No other files are affected\n"
        f"  - Explain the expected benefit\n"
        f"  - Rate impact: HIGH / MEDIUM / LOW\n"
        f"  - Wrap any example code in: === CODE BEGIN: <filename> === ... === CODE END: <filename> ===\n\n"
        f"Do not use markdown formatting."
    )
    print("[Assistant] Analysing system for improvement opportunities...")
    result = llm(prompt)
    _record("system", f"Self-improvement analysis: {query}", applied=False)
    return result


def _handle_troubleshoot(query: str, llm) -> str:
    core_files = [
        "legal_faiss.py", "legal_distiller.py", "intent_router.py",
        "reasoning_engine.py", "ingest.py", "app.py"
    ]
    context_parts = []
    for f in core_files:
        content = _read_file(f)
        if content:
            context_parts.append(f"=== FILE: {f} ===\n{content}\n=== END FILE ===")

    all_context = "\n\n".join(context_parts)

    print("[Assistant] Scanning project entities...")
    live_ctx = _build_live_context(PROJECT_ROOT)

    prompt = (
        f"{SYSTEM_CONTEXT}\n\n"
        f"{live_ctx}\n\n"
        f"The user is reporting this problem:\n{query}\n\n"
        f"Here are the relevant project files:\n\n{all_context}\n\n"
        f"Diagnose the problem. Identify the root cause and which file contains it.\n"
        f"For each fix you suggest:\n"
        f"  IMPACT ASSESSMENT: Which other pipeline files are affected by that fix.\n"
        f"  REQUIRED ALIGNED CHANGES: Exactly what must change in those files to keep the system consistent.\n"
        f"  If a fix affects no other files, explicitly state: No other files are affected.\n"
        f"  Wrap the fix code in: === CODE BEGIN: <filename> === ... === CODE END: <filename> ===\n\n"
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
