import os
import re
import json
import uuid
import datetime
import threading
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

from legal_faiss import LegalFAISS
from legal_distiller import distil, extract_case_name
from intent_router import route
from reasoning_engine import reason
from formatter import format_response, format_no_results, format_error

# Optional tool imports — fail gracefully if not installed
try:
    from statute_lookup_tool import lookup_statute
    STATUTE_TOOL = True
except ImportError:
    STATUTE_TOOL = False

try:
    from treatment_analyzer import analyse_treatment
    TREATMENT_TOOL = True
except ImportError:
    TREATMENT_TOOL = False

try:
    from commonlii_tool import search_and_download as commonlii_search
    COMMONLII_TOOL = True
except ImportError:
    COMMONLII_TOOL = False
    commonlii_search = None

try:
    from elitigation_tool import search_and_download as elitigation_search
    ELITIGATION_TOOL = True
except ImportError:
    ELITIGATION_TOOL = False
    elitigation_search = None

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)
rag = LegalFAISS()

# Absolute path to the directory containing this file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# In-memory store for background research jobs
RESEARCH_JOBS = {}

# ---------------------------------------------------------------------------
# Shared project context — used by both /agent and /gemini endpoints
# ---------------------------------------------------------------------------

SYSTEM_CONTEXT = """You are an expert Python developer assisting with a Singapore Legal AI system.

DO NOT RUSH. Think carefully and thoroughly before responding. Consider the full consequences
of every suggestion before writing a single line of output.

Project architecture:
- LLM: llama3.1 via Ollama (http://localhost:11434/api/generate)
- Embeddings: nomic-embed-text via Ollama
- Vector store: data/cases_db.json (key is "text", not "chunk")
- Query pipeline: intent_router.py -> app.py -> legal_faiss.py -> legal_distiller.py -> reasoning_engine.py -> formatter.py
- REST API: api.py (Flask, port 5000)
- Coding assistant: legal_assistant.py

Coding conventions:
- British spelling in all comments, docstrings, and print statements
- No markdown formatting in LLM output (plain text only)
- Atomic file saves: write to .tmp file first, then os.replace() to final path
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
value already used by this project, as confirmed in the LIVE PROJECT ENTITIES section below.
If you are unsure of a value, say so explicitly in plain text BEFORE the code — do not guess.

CRITICAL RULE — CODE DELIMITERS:
Whenever you output code (whether a full file or a snippet), you MUST wrap it exactly as follows:

=== CODE BEGIN: <filename> ===
<code here>
=== CODE END: <filename> ===

This applies to every piece of code in your response without exception.
"""


# ---------------------------------------------------------------------------
# Entity scanner — reads project files at startup to extract live values
# ---------------------------------------------------------------------------

def _scan_project_entities():
    """Scan all .py files in PROJECT_ROOT and extract key entities.

    Returns a dict of discovered values covering: LLM models, URLs, DB paths,
    embedding models, class names, and inter-file function references.
    This is injected into every /agent prompt so the LLM never guesses.
    """
    entities = {
        "llm_models":       [],
        "embedding_models": [],
        "ollama_urls":      [],
        "db_paths":         [],
        "db_keys":          [],
        "flask_port":       None,
        "classes":          {},   # filename -> [class names]
        "functions":        {},   # filename -> [top-level function names]
        "imports":          {},   # filename -> [imported module names]
        "file_list":        [],
    }

    py_files = sorted(
        f for f in os.listdir(PROJECT_ROOT)
        if f.endswith(".py") and os.path.isfile(os.path.join(PROJECT_ROOT, f))
    )
    entities["file_list"] = py_files

    for fname in py_files:
        path = os.path.join(PROJECT_ROOT, fname)
        try:
            src = open(path, "r", encoding="utf-8").read()
        except OSError:
            continue

        # LLM model names (e.g. "llama3.1", "gemini-2.5-flash", "nomic-embed-text")
        for m in re.findall(r'["\']([a-zA-Z0-9][a-zA-Z0-9.:\-_/]+)["\']', src):
            if any(kw in m.lower() for kw in ("llama", "gemini", "gpt", "claude", "mistral", "phi", "nomic", "embed")):
                target = entities["llm_models"] if "embed" not in m.lower() else entities["embedding_models"]
                if m not in target:
                    target.append(m)

        # Ollama / API URLs
        for u in re.findall(r'["\']https?://[^"\' ]+["\']', src):
            u = u.strip("'\"")
            if u not in entities["ollama_urls"]:
                entities["ollama_urls"].append(u)

        # DB / data file paths
        for p in re.findall(r'["\']data/[^"\' ]+\.json["\']', src):
            p = p.strip("'\"")
            if p not in entities["db_paths"]:
                entities["db_paths"].append(p)

        # DB dict keys used in vector store entries
        for k in re.findall(r'\[[\'"](text|chunk|source|folder|vector)[\'"]\]', src):
            if k not in entities["db_keys"]:
                entities["db_keys"].append(k)

        # Flask port
        port_match = re.search(r'port\s*=\s*(\d+)', src)
        if port_match and entities["flask_port"] is None:
            entities["flask_port"] = port_match.group(1)

        # Class names
        classes = re.findall(r'^class\s+(\w+)', src, re.MULTILINE)
        if classes:
            entities["classes"][fname] = classes

        # Top-level function names
        funcs = re.findall(r'^def\s+(\w+)', src, re.MULTILINE)
        if funcs:
            entities["functions"][fname] = funcs

        # Imported module names
        imps = re.findall(r'^(?:import|from)\s+(\S+)', src, re.MULTILINE)
        if imps:
            entities["imports"][fname] = imps

    return entities


def _build_live_context():
    """Return a formatted plain-text block of live project entities.

    This is prepended to every /agent prompt so the LLM always works
    from actual values, never from assumptions or placeholders.
    """
    e = _scan_project_entities()

    lines = [
        "LIVE PROJECT ENTITIES (scanned from disk — use these exact values, never placeholders):",
        "",
    ]

    lines.append(f"Project files present: {', '.join(e['file_list']) if e['file_list'] else '(none found)'}")

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

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _detect_statute_ref(q):
    """Return True if the query appears to reference a named statute."""
    statutes = [
        "penal code", "mda", "misuse of drugs", "cpc",
        "criminal procedure code", "evidence act",
        "prevention of corruption", "arms act", "computer misuse",
    ]
    return any(s in q.lower() for s in statutes)


def _detect_treatment_query(q):
    """Return True if the query asks about how a case was treated."""
    keywords = [
        "treatment of", "treated in", "cited in",
        "followed in", "distinguished in", "overruled",
    ]
    return any(k in q.lower() for k in keywords)


def _read_project_file(filename):
    """Read a .py file from PROJECT_ROOT and return its contents as a string.

    Returns None if the file cannot be found or read.
    """
    filepath = os.path.join(PROJECT_ROOT, filename)
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def _call_ollama(prompt):
    """Send a prompt to the local Ollama instance and return the response text.

    Raises requests.RequestException on network errors.
    """
    payload = {
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _call_agent_llm(prompt):
    """Call Gemini 2.5 Flash for the /agent endpoint.

    Falls back to Ollama llama3.1 if GEMINI_API_KEY is not set or
    google-generativeai is not installed.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            resp  = model.generate_content(prompt)
            return resp.text.strip()
        except ImportError:
            pass  # google-generativeai not installed — fall through to Ollama
        except Exception:
            pass  # Gemini call failed — fall through to Ollama

    # Fallback: local Ollama
    return _call_ollama(prompt)


# ---------------------------------------------------------------------------
# Existing endpoints (unchanged)
# ---------------------------------------------------------------------------

@app.route("/query", methods=["POST"])
def query():
    """Accept a legal query and return a reasoned response with sources."""
    data = request.get_json(silent=True) or {}
    q    = (data.get("query") or "").strip()
    hint = (data.get("mode") or "").strip().lower()

    if not q:
        return jsonify({"error": "No query provided"}), 400

    if hint and hint in ("irac", "case_summary", "synthesis", "sentencing", "elements", "procedure", "drafting"):
        mode = hint
    else:
        mode = route(q)

    if mode == "case_summary":
        case_name  = extract_case_name(q)
        raw_chunks = rag.search_by_source(q, case_name, top_k=10) if case_name else rag.search(q, top_k=8)
    else:
        raw_chunks = rag.search(q, top_k=8)

    chunks = distil(q, raw_chunks, top_k=6)

    if not chunks:
        return jsonify({
            "response":  format_no_results(q),
            "sources":   [],
            "mode":      mode,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        })

    sources = list(dict.fromkeys(
        c.get("source") or c.get("meta", {}).get("source", "Unknown")
        for c in chunks
    ))

    supplement = ""

    if STATUTE_TOOL and _detect_statute_ref(q):
        try:
            s = lookup_statute(q)
            if s:
                supplement += f"\n[Statute]\n{s}\n"
        except Exception:
            pass

    if TREATMENT_TOOL and _detect_treatment_query(q):
        try:
            r = analyse_treatment(q)
            if r and r.get("total", 0) > 0:
                from treatment_analyzer import format_treatment_report
                supplement += "\n" + format_treatment_report(r)
        except Exception:
            pass

    try:
        response = reason(q, chunks, mode=mode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if supplement:
        response = response + "\n\n" + supplement.strip()

    return jsonify({
        "response":  response,
        "sources":   sources,
        "mode":      mode,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    })


@app.route("/sources", methods=["GET"])
def sources():
    """Return a list of all ingested source documents."""
    return jsonify({"sources": rag.list_sources()})


@app.route("/health", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    gemini_key_set = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    return jsonify({
        "status":     "ok",
        "vectors":    rag.count(),
        "gemini_key": "ok" if gemini_key_set else "missing",
    })


# ---------------------------------------------------------------------------
# New endpoint 1: /agent — Python coding assistant
# ---------------------------------------------------------------------------

@app.route("/agent", methods=["POST"])
def agent():
    """Coding assistant endpoint.

    Accepts a free-text command and delegates to the local Ollama LLM
    (llama3.1) to review, rewrite, troubleshoot, or advise on project files.

    Request body
    ------------
    {"command": "review ingest.py"}

    Response
    --------
    {"response": "...", "timestamp": "..."}
    """
    try:
        data    = request.get_json(silent=True) or {}
        command = (data.get("command") or "").strip()

        if not command:
            return jsonify({
                "response":  "No command provided.",
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }), 400

        cmd_lower = command.lower()

        # Scan project files to build live entity context for this request
        live_ctx = _build_live_context()

        # ── list / show files ──────────────────────────────────────────────
        if cmd_lower in ("list files", "show files"):
            py_files = sorted(
                f for f in os.listdir(PROJECT_ROOT)
                if f.endswith(".py") and os.path.isfile(os.path.join(PROJECT_ROOT, f))
            )
            result = "Python files in project:\n" + "\n".join(py_files)

        # ── history ───────────────────────────────────────────────────────
        elif cmd_lower == "history":
            history_path = os.path.join(PROJECT_ROOT, "data", "code_improvements.json")
            if not os.path.isfile(history_path):
                result = "No improvement history found (data/code_improvements.json does not exist)."
            else:
                try:
                    with open(history_path, "r", encoding="utf-8") as fh:
                        entries = json.load(fh)
                    last_20 = entries[-20:] if isinstance(entries, list) else [entries]
                    result  = json.dumps(last_20, indent=2, ensure_ascii=False)
                except Exception as hist_err:
                    result = f"Could not read history: {hist_err}"

        # ── review / check / audit ────────────────────────────────────────
        elif cmd_lower.startswith(("review ", "check ", "audit ")):
            parts    = command.split(None, 1)
            filename = parts[1].strip() if len(parts) > 1 else ""
            if not filename.endswith(".py"):
                filename += ".py"
            content = _read_project_file(filename)
            if content is None:
                result = f"File not found in project: {filename}"
            else:
                prompt = (
                    f"{SYSTEM_CONTEXT}\n\n"
                    f"{live_ctx}\n\n"
                    f"Please review the following file ({filename}) for bugs, interface mismatches "
                    f"with the rest of the pipeline, and potential improvements. "
                    f"Be specific and reference line numbers where possible. "
                    f"For each issue found, state: (1) IMPACT ASSESSMENT — which other pipeline files are affected; "
                    f"(2) REQUIRED ALIGNED CHANGES — what must change in those files to stay consistent. "
                    f"If no other files are affected by an issue, explicitly state so. "
                    f"Use plain text only — no markdown.\n\n"
                    f"=== FILE: {filename} ===\n{content}\n=== END FILE ==="
                )
                result = _call_agent_llm(prompt)

        # ── rewrite ───────────────────────────────────────────────────────
        elif cmd_lower.startswith("rewrite "):
            parts    = command.split(None, 1)
            rest     = parts[1].strip() if len(parts) > 1 else ""
            tokens   = rest.split(None, 1)
            filename = tokens[0].strip()
            improvement_desc = tokens[1].strip() if len(tokens) > 1 else "improve code quality"
            if not filename.endswith(".py"):
                filename += ".py"
            content = _read_project_file(filename)
            if content is None:
                result = f"File not found in project: {filename}"
            else:
                prompt = (
                    f"{SYSTEM_CONTEXT}\n\n"
                    f"{live_ctx}\n\n"
                    f"Rewrite the following file ({filename}) with this improvement: {improvement_desc}\n\n"
                    f"BEFORE providing the rewritten code, first produce a plain-text IMPACT ASSESSMENT: "
                    f"list every other file in the pipeline that will need a corresponding change as a result of this rewrite, "
                    f"and state exactly what must change in each. If no other files are affected, explicitly state so.\n\n"
                    f"Then return the complete rewritten Python source code wrapped in delimiters:\n"
                    f"=== CODE BEGIN: {filename} ===\n<code>\n=== CODE END: {filename} ===\n\n"
                    f"=== FILE: {filename} ===\n{content}\n=== END FILE ==="
                )
                result = _call_agent_llm(prompt)

        # ── troubleshoot / error / problem ────────────────────────────────
        elif cmd_lower.startswith(("troubleshoot", "error", "problem")):
            core_files = [
                "legal_faiss.py",
                "legal_distiller.py",
                "intent_router.py",
                "reasoning_engine.py",
                "ingest.py",
                "app.py",
            ]
            file_sections = []
            for fname in core_files:
                content = _read_project_file(fname)
                if content:
                    file_sections.append(f"=== FILE: {fname} ===\n{content}\n=== END FILE ===")
            combined = "\n\n".join(file_sections) if file_sections else "(no core files found)"
            prompt = (
                f"{SYSTEM_CONTEXT}\n\n"
                f"{live_ctx}\n\n"
                f"The user reports: {command}\n\n"
                f"Please diagnose the most likely causes of this problem by examining the core "
                f"project files below. For each fix you suggest: "
                f"(1) state the IMPACT ASSESSMENT — which other pipeline files are affected by that fix; "
                f"(2) state the REQUIRED ALIGNED CHANGES needed in those files to keep the system consistent. "
                f"If a fix affects no other files, explicitly state so. "
                f"Wrap any code fixes in: === CODE BEGIN: <filename> === ... === CODE END: <filename> ===\n"
                f"Use plain text only — no markdown.\n\n"
                f"{combined}"
            )
            result = _call_agent_llm(prompt)

        # ── general improvement / suggestion / other ───────────────────────
        else:
            prompt = (
                f"{SYSTEM_CONTEXT}\n\n"
                f"{live_ctx}\n\n"
                f"The user asks: {command}\n\n"
                f"Provide a thoughtful, specific response relevant to the Singapore Legal AI project. "
                f"If your response involves any code change or suggestion, first state the IMPACT ASSESSMENT "
                f"(which other pipeline files are affected) and any REQUIRED ALIGNED CHANGES needed. "
                f"If no other files are affected, explicitly state so. "
                f"Wrap any code in: === CODE BEGIN: <filename> === ... === CODE END: <filename> ===\n"
                f"Use plain text only — no markdown."
            )
            result = _call_agent_llm(prompt)

        return jsonify({
            "response":  result,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        })

    except Exception as e:
        return jsonify({
            "response":  f"Agent error: {e}",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }), 500


# ---------------------------------------------------------------------------
# New endpoint 2: /gemini — Gemini API passthrough
# ---------------------------------------------------------------------------

@app.route("/gemini", methods=["POST"])
def gemini():
    """Forward a message to the Gemini API and return the response.

    Requires the GEMINI_API_KEY environment variable to be set.
    Install dependency: pip install google-generativeai

    Request body
    ------------
    {"message": "explain ingest.py"}

    Response
    --------
    {"response": "...", "timestamp": "..."}
    """
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify({
            "response":  "GEMINI_API_KEY not set. Set it as an environment variable and restart the server.",
            "timestamp": timestamp,
        }), 400

    # Check for google-generativeai package
    try:
        import google.generativeai as genai  # pip install google-generativeai
    except ImportError:
        return jsonify({
            "response":  "google-generativeai not installed. Run: pip install google-generativeai",
            "timestamp": timestamp,
        }), 400

    try:
        data    = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()

        if not message:
            return jsonify({
                "response":  "No message provided.",
                "timestamp": timestamp,
            }), 400

        genai.configure(api_key=api_key)
        model    = genai.GenerativeModel("gemini-2.5-flash")
        full_prompt = SYSTEM_CONTEXT + "\n\nUser: " + message
        resp     = model.generate_content(full_prompt)

        return jsonify({
            "response":  resp.text,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        })

    except Exception as e:
        return jsonify({
            "response":  f"Gemini error: {e}",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }), 500


# ---------------------------------------------------------------------------
# New endpoint 3: POST /research/background — spawn background research job
# ---------------------------------------------------------------------------

def _run_research_job(job_id, query):
    """Background worker: search CommonLII and eLitigation, then update job state."""
    saved_files = []
    any_success = False

    # CommonLII search
    if COMMONLII_TOOL and commonlii_search is not None:
        try:
            results = commonlii_search(query, max_results=5, max_download=3)
            if results:
                saved_files.extend(results)
                any_success = True
        except Exception as commonlii_err:
            print(f"[research] CommonLII error for job {job_id}: {commonlii_err}")

    # eLitigation search
    if ELITIGATION_TOOL and elitigation_search is not None:
        try:
            results = elitigation_search(query, max_results=5, max_download=3)
            if results:
                saved_files.extend(results)
                any_success = True
        except Exception as elit_err:
            print(f"[research] eLitigation error for job {job_id}: {elit_err}")

    status = "complete" if any_success else "failed"

    RESEARCH_JOBS[job_id] = {
        "status":    status,
        "files":     saved_files,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    print(f"[research] Job {job_id} finished with status '{status}', {len(saved_files)} file(s) saved.")


@app.route("/research/background", methods=["POST"])
def research_background():
    """Trigger a background research job against CommonLII and eLitigation.

    Returns immediately with a job_id that can be polled via GET /research/status/<job_id>.

    Request body
    ------------
    {"query": "drug trafficking sentencing"}

    Response
    --------
    {"job_id": "abc123", "status": "started"}
    """
    data  = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    job_id = uuid.uuid4().hex[:8]
    RESEARCH_JOBS[job_id] = {"status": "running", "files": [], "timestamp": None}

    thread = threading.Thread(
        target=_run_research_job,
        args=(job_id, query),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "started"})


# ---------------------------------------------------------------------------
# New endpoint 4: GET /research/status/<job_id> — poll research job status
# ---------------------------------------------------------------------------

@app.route("/research/status/<job_id>", methods=["GET"])
def research_status(job_id):
    """Return the current status of a background research job.

    Response
    --------
    {"job_id": "...", "status": "running"|"complete"|"failed", "files": [...], "timestamp": "..."}
    Returns 404 if the job_id is not recognised.
    """
    job = RESEARCH_JOBS.get(job_id)
    if job is None:
        return jsonify({"status": "not_found"}), 404

    return jsonify({
        "job_id":    job_id,
        "status":    job.get("status", "unknown"),
        "files":     job.get("files", []),
        "timestamp": job.get("timestamp"),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Legal AI — REST API Server")
    print(f"  Loaded {rag.count()} vectors from database")
    print("  Running at http://localhost:5000")
    print("  Open legal-ai-ui/index.html in your browser")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
