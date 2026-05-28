# FILE: app.py
# LOCATION: C:\Users\Admin\Desktop\ai-agent-system\app.py
# ACTION: Replace entire file

import json
import os
from legal_faiss import LegalFAISS
from legal_distiller import distil, extract_case_name
from intent_router import route
from reasoning_engine import reason
from formatter import format_response, format_no_results, format_error

# ── TOOL IMPORTS ──────────────────────────────────────────────────────────────
try:
    from statute_lookup_tool import lookup_statute
    STATUTE_TOOL = True
except ImportError:
    STATUTE_TOOL = False

try:
    from treatment_analyzer import analyse_treatment, format_treatment_report
    TREATMENT_TOOL = True
except ImportError:
    TREATMENT_TOOL = False

try:
    from research_agent import research
    RESEARCH_TOOL = True
except ImportError:
    RESEARCH_TOOL = False

# ── INIT ──────────────────────────────────────────────────────────────────────
rag = LegalFAISS()

# ── AUTO TOOL SELECTION ───────────────────────────────────────────────────────

def detect_statute_ref(query):
    """Return True if the query appears to reference a named statute."""
    statutes = [
        "penal code", "pc", "mda", "misuse of drugs",
        "cpc", "criminal procedure code", "evidence act",
        "mla", "money laundering", "corruption",
        "pca", "prevention of corruption",
        "arms offences", "arms act",
        "computer misuse", "cma",
    ]
    q = query.lower()
    return any(s in q for s in statutes)


def detect_treatment_query(query):
    """Return True if the query asks about how a case was treated."""
    keywords = [
        "treatment of", "treated in", "cited in",
        "followed in", "distinguished in", "overruled",
        "positive treatment", "negative treatment",
        "how has", "subsequent cases",
    ]
    q = query.lower()
    return any(k in q for k in keywords)


def detect_research_query(query):
    """Return True if the query requests external case research."""
    keywords = [
        "find cases", "search for", "look up",
        "are there any cases", "what cases",
        "recent cases", "latest cases",
        "commonlii", "elitigation",
    ]
    q = query.lower()
    return any(k in q for k in keywords)


# ── MEMORY HELPERS ────────────────────────────────────────────────────────────

CHAT_HISTORY_PATH = "data/chats.json"


def load_chat_history():
    """Load chat history from disk.

    Returns an empty list if the file does not exist or is corrupted.
    Corrupt files are silently discarded rather than crashing the session.
    """
    if not os.path.exists(CHAT_HISTORY_PATH):
        return []
    try:
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_chat_history(history):
    """Save chat history atomically.

    Writes to a temporary file first, then uses os.replace() to swap it in.
    This prevents corruption if the process is interrupted mid-write.
    """
    os.makedirs("data", exist_ok=True)
    tmp_path = CHAT_HISTORY_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, CHAT_HISTORY_PATH)


# ── LIST SOURCES HELPER ───────────────────────────────────────────────────────

def list_sources():
    """Return a deduplicated list of all source filenames in the database.

    Uses the internal _db list on the LegalFAISS instance directly,
    since LegalFAISS.query() is the only public retrieval method.
    """
    try:
        db = rag._db
        seen = []
        for entry in db:
            src = entry.get("source", "")
            if src and src not in seen:
                seen.append(src)
        return seen
    except Exception:
        return []


# ── MAIN CHAT LOOP ────────────────────────────────────────────────────────────

def chat():
    print("\n" + "=" * 55)
    print("  Legal AI — Singapore Law Assistant")
    print("  Type 'exit' to quit | 'history' to review chat")
    print("  Type 'sources' to list all ingested cases")
    print("=" * 55 + "\n")

    history = load_chat_history()

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Exiting]")
            break

        if not query:
            continue

        if query.lower() == "exit":
            print("[Session ended]")
            break

        if query.lower() == "history":
            if not history:
                print("[No chat history yet]\n")
            else:
                for h in history[-5:]:
                    print(f"\nYou: {h['query']}")
                    print(f"AI:  {h['response'][:300]}...")
            continue

        if query.lower() == "sources":
            sources = list_sources()
            if sources:
                print(f"\n[{len(sources)} sources in database]")
                for s in sources:
                    print(f"  {s}")
                print()
            else:
                print("[No sources ingested yet — run ingest.py first]\n")
            continue

        # ── ROUTE ─────────────────────────────────────────────────────────────
        # intent_router.route() returns a plain string.
        # detect_area_of_law is handled inside the router itself for api.py;
        # in the CLI we pass area=None to let the vector store search globally.
        mode = route(query)
        print(f"\n[Mode: {mode.upper()}]")

        # ── AUTO TOOL: STATUTE LOOKUP ──────────────────────────────────────────
        if STATUTE_TOOL and detect_statute_ref(query):
            print("[Tool: Statute Lookup]")
            try:
                statute_result = lookup_statute(query)
                if statute_result:
                    print("\n── Statute ──────────────────────────────────────")
                    print(statute_result)
                    print("─────────────────────────────────────────────────\n")
            except Exception as e:
                print(f"[Statute lookup failed: {e}]")

        # ── AUTO TOOL: TREATMENT ANALYSIS ─────────────────────────────────────
        if TREATMENT_TOOL and detect_treatment_query(query):
            print("[Tool: Treatment Analyser]")
            try:
                treatment_result = analyse_treatment(query)
                if treatment_result and treatment_result.get("total", 0) > 0:
                    print("\n── Treatment Analysis ───────────────────────────")
                    print(format_treatment_report(treatment_result))
                    print("─────────────────────────────────────────────────\n")
            except Exception as e:
                print(f"[Treatment analysis failed: {e}]")

        # ── AUTO TOOL: RESEARCH AGENT ─────────────────────────────────────────
        if RESEARCH_TOOL and detect_research_query(query):
            print("[Tool: Research Agent — this will search external sources]")
            confirm = input("Proceed with external search? (y/n): ").strip().lower()
            if confirm == "y":
                try:
                    research_result = research(query)
                    if research_result:
                        print("\n── Research Results ─────────────────────────────")
                        print(research_result)
                        print("─────────────────────────────────────────────────\n")
                except Exception as e:
                    print(f"[Research agent failed: {e}]")

        # ── RETRIEVAL ─────────────────────────────────────────────────────────
        # LegalFAISS.query() is the only retrieval method.
        # For case_summary mode, we still call query() — the case name extracted
        # by extract_case_name() is printed for user feedback only; the area
        # filter in query() is the correct narrowing mechanism.
        if mode == "case_summary":
            case_name = extract_case_name(query)
            if case_name:
                print(f"[Retrieval: searching for '{case_name}']")
        raw_chunks = rag.query(query)

        # ── DISTILLATION ──────────────────────────────────────────────────────
        # distil() accepts exactly one argument: the list of raw chunks.
        chunks = distil(raw_chunks)

        if not chunks:
            print(format_no_results(query))
            continue

        # ── SOURCES ───────────────────────────────────────────────────────────
        sources = list(dict.fromkeys(
            c.get("source", "Unknown") for c in chunks
        ))

        # ── REASONING ─────────────────────────────────────────────────────────
        # reason(query, intent, chunks) — intent is the second positional argument.
        # mode is used directly as the intent string (both are the same routing token).
        try:
            response = reason(query, mode, chunks)
        except Exception as e:
            print(format_error(str(e)))
            continue

        # ── FORMAT + DISPLAY ──────────────────────────────────────────────────
        formatted = format_response(response, mode=mode, sources=sources, query=query)
        print(formatted)

        # ── SAVE TO HISTORY ───────────────────────────────────────────────────
        history.append({"query": query, "response": response})
        save_chat_history(history)


if __name__ == "__main__":
    chat()
