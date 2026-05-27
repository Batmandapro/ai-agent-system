import os
import json
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from legal_faiss import LegalFAISS
from legal_distiller import distil, extract_case_name
from intent_router import route
from reasoning_engine import reason
from formatter import format_response, format_no_results, format_error

# ── FLASK SETUP ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow the UI (file://) to call this server

rag = LegalFAISS()

# ── TOOL IMPORTS ──────────────────────────────────────────────────────────────
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

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _detect_statute_ref(q):
    statutes = [
        "penal code", "mda", "misuse of drugs", "cpc",
        "criminal procedure code", "evidence act",
        "prevention of corruption", "arms act", "computer misuse"
    ]
    return any(s in q.lower() for s in statutes)

def _detect_treatment_query(q):
    keywords = [
        "treatment of", "treated in", "cited in",
        "followed in", "distinguished in", "overruled"
    ]
    return any(k in q.lower() for k in keywords)

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/query", methods=["POST"])
def query():
    """
    Main query endpoint. Called by the UI.

    Request body: {"query": "...", "mode": "..."}
    Response:     {"response": "...", "sources": [...], "mode": "...", "timestamp": "..."}
    """
    data  = request.get_json(silent=True) or {}
    q     = (data.get("query") or "").strip()
    hint  = (data.get("mode") or "").strip().lower()

    if not q:
        return jsonify({"error": "No query provided"}), 400

    # Use the hinted mode from the UI if provided, else auto-route
    if hint and hint in (
        "irac", "case_summary", "synthesis", "sentencing",
        "elements", "procedure", "drafting"
    ):
        mode = hint
    else:
        mode = route(q)

    # ── RETRIEVAL ─────────────────────────────────────────────────────────────
    if mode == "case_summary":
        case_name = extract_case_name(q)
        if case_name:
            raw_chunks = rag.search_by_source(q, case_name, top_k=10)
        else:
            raw_chunks = rag.search(q, top_k=8)
    else:
        raw_chunks = rag.search(q, top_k=8)

    chunks = distil(q, raw_chunks, top_k=6)

    if not chunks:
        return jsonify({
            "response":  format_no_results(q),
            "sources":   [],
            "mode":      mode,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
        })

    sources = list(dict.fromkeys(
        c.get("source") or c.get("meta", {}).get("source", "Unknown")
        for c in chunks
    ))

    # ── SUPPLEMENTARY TOOLS ───────────────────────────────────────────────────
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

    # ── REASONING ─────────────────────────────────────────────────────────────
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
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds")
    })

@app.route("/sources", methods=["GET"])
def sources():
    """Return all ingested sources."""
    return jsonify({"sources": rag.list_sources()})

@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({"status": "ok", "vectors": rag.count()})

# ── STARTUP ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Legal AI — REST API Server")
    print(f"  Loaded {rag.count()} vectors from database")
    print("  Running at http://localhost:5000")
    print("  Open legal-ai-ui/index.html in your browser")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)