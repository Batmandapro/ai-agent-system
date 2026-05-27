import os
from intent_router import route
from legal_distiller import distil
from reasoning_engine import reason
from formatter import format_response, format_error, format_no_results
from treatment_analyzer import analyze_treatment, format_treatment_report

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Controls how many cases to download from each free source per research session
MAX_COMMONLII_RESULTS  = 10
MAX_COMMONLII_DOWNLOAD = 5
MAX_ELITIGATION_RESULTS  = 10
MAX_ELITIGATION_DOWNLOAD = 5

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _prompt_user_confirmation(message: str) -> bool:
    """
    Pause and ask the user to confirm before proceeding.
    Returns True if confirmed, False if declined.
    """
    print(f"\n[AGENT] {message}")
    choice = input("  Proceed? (y/n): ").strip().lower()
    return choice in ("y", "yes")

def _re_ingest():
    """
    Re-run ingestion to embed any newly downloaded cases.
    Called automatically after downloading from online sources.
    """
    print("\n[AGENT] Re-ingesting newly downloaded cases...")
    import ingest
    ingest.main()

# ── RESEARCH PIPELINE ─────────────────────────────────────────────────────────

def _search_local(query: str, mode: str) -> dict:
    """Step 1 — Search the local vector store first."""
    print(f"\n[AGENT] Step 1: Searching local database...")
    result = distil(query, mode)
    print(f"  Found {result['chunk_count']} relevant chunks locally")
    return result

def _search_free_sources(query: str) -> list:
    """
    Step 2 — Search CommonLII and eLitigation for additional cases.
    Downloads and saves them to data/cases/ for re-ingestion.
    Returns list of saved file paths.
    """
    saved = []

    # CommonLII
    try:
        from commonlii_tool import search_and_download as commonlii_search
        print(f"\n[AGENT] Step 2a: Searching CommonLII...")
        paths = commonlii_search(
            query,
            max_results=MAX_COMMONLII_RESULTS,
            max_download=MAX_COMMONLII_DOWNLOAD
        )
        saved.extend(paths)
    except Exception as e:
        print(f"  [SKIP] CommonLII unavailable: {e}")

    # eLitigation
    try:
        from elitigation_tool import search_and_download as elitigation_search
        print(f"\n[AGENT] Step 2b: Searching eLitigation...")
        paths = elitigation_search(
            query,
            max_results=MAX_ELITIGATION_RESULTS,
            max_download=MAX_ELITIGATION_DOWNLOAD
        )
        saved.extend(paths)
    except Exception as e:
        print(f"  [SKIP] eLitigation unavailable: {e}")

    return saved

def _search_lawnet(query: str) -> list:
    """
    Step 3 — Supervised Lawnet session.
    Only called if local + free sources yield insufficient results.
    Requires user to be present and logged in.
    """
    try:
        from lawnet_tool import supervised_search
        print(f"\n[AGENT] Step 3: Falling back to Lawnet (supervised session)...")
        return supervised_search(query)
    except ImportError:
        print(f"  [SKIP] lawnet_tool.py not yet installed")
        return []
    except Exception as e:
        print(f"  [SKIP] Lawnet session failed: {e}")
        return []

# ── PUBLIC API ────────────────────────────────────────────────────────────────

def research(query: str, use_online: bool = True, use_lawnet: bool = False) -> str:
    """
    Full research pipeline:
    1. Search local vector store
    2. If insufficient — search CommonLII + eLitigation (free)
    3. Re-ingest downloaded cases
    4. Re-search local store with new cases
    5. If still insufficient and use_lawnet=True — supervised Lawnet session
    6. Reason over all retrieved context
    7. Return formatted research memo

    Args:
        query:       Research question or area of law
        use_online:  Whether to search free online sources (default True)
        use_lawnet:  Whether to fall back to supervised Lawnet session (default False)
    """

    mode = route(query)
    print(f"\n[AGENT] Research mode: {mode.upper()}")
    print(f"[AGENT] Query: {query}")
    print("=" * 60)

    # Step 1 — Local search
    local_result = _search_local(query, mode)

    # Step 2 — Supplement with free online sources if needed
    new_files = []
    if use_online and local_result["chunk_count"] < 3:
        print(f"\n[AGENT] Local results insufficient ({local_result['chunk_count']} chunks).")
        if _prompt_user_confirmation("Search CommonLII and eLitigation for additional cases?"):
            new_files = _search_free_sources(query)

            if new_files:
                print(f"\n[AGENT] Downloaded {len(new_files)} new case(s). Re-ingesting...")
                _re_ingest()
                # Re-search with newly ingested cases
                local_result = _search_local(query, mode)

    # Step 3 — Lawnet fallback
    if use_lawnet and local_result["chunk_count"] < 2:
        print(f"\n[AGENT] Still insufficient results ({local_result['chunk_count']} chunks).")
        if _prompt_user_confirmation("Fall back to supervised Lawnet session?"):
            lawnet_files = _search_lawnet(query)
            if lawnet_files:
                _re_ingest()
                local_result = _search_local(query, mode)

    # Step 4 — Handle truly empty results
    if local_result["chunk_count"] == 0:
        return format_no_results(query)

    # Step 5 — Reason over retrieved context
    print(f"\n[AGENT] Reasoning over {local_result['chunk_count']} chunks from "
          f"{len(local_result['sources'])} source(s)...")

    raw_answer = reason(
        query=query,
        context_chunks=local_result["raw_chunks"],
        mode=mode
    )

    # Step 6 — Format and return
    response = format_response(
        raw=raw_answer,
        mode=mode,
        sources=local_result["sources"],
        query=query
    )

    # Step 7 — Append download summary if new cases were found
    if new_files:
        response += f"\n  [New cases downloaded and ingested: {len(new_files)}]\n"
        for f in new_files:
            response += f"    • {os.path.basename(f)}\n"

    return response


def research_with_treatment(query: str, target_case: str, **kwargs) -> str:
    """
    Run a full research query AND append a treatment analysis
    for a specific target case.

    Useful when you want to know both the law on an issue
    AND how a particular case has been treated subsequently.
    """
    research_output  = research(query, **kwargs)
    treatment_report = analyze_treatment(target_case)
    treatment_output = format_treatment_report(treatment_report)

    return research_output + "\n" + treatment_output


# ── INTERACTIVE MODE ──────────────────────────────────────────────────────────

BANNER = """
============================================================
  Legal Research Agent
  Commands:
    research <query>          — Full research with online sources
    treatment <case name>     — Case treatment analysis
    local <query>             — Local database only (no download)
    lawnet <query>            — Include supervised Lawnet session
    exit                      — Quit
============================================================
"""

def run_interactive():
    print(BANNER)

    while True:
        try:
            user_input = input("Research: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye.")
            break

        if not user_input:
            continue

        parts   = user_input.split(" ", 1)
        command = parts[0].lower()
        arg     = parts[1] if len(parts) > 1 else ""

        if command in ("exit", "quit", "q"):
            print("Exiting. Goodbye.")
            break

        elif command == "research" and arg:
            print(research(arg, use_online=True, use_lawnet=False))

        elif command == "local" and arg:
            print(research(arg, use_online=False, use_lawnet=False))

        elif command == "lawnet" and arg:
            print(research(arg, use_online=True, use_lawnet=True))

        elif command == "treatment" and arg:
            report = analyze_treatment(arg)
            print(format_treatment_report(report))

        else:
            print("  Unrecognised command. Type 'exit' to quit.")


if __name__ == "__main__":
    run_interactive()