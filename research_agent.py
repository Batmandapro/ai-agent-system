import os
from intent_router import route
from legal_faiss import LegalFAISS
from legal_distiller import distil, extract_case_name
from reasoning_engine import reason
from formatter import format_response, format_error, format_no_results
from treatment_analyzer import analyse_treatment, format_treatment_report

MAX_COMMONLII_RESULTS   = 10
MAX_COMMONLII_DOWNLOAD  = 5
MAX_ELITIGATION_RESULTS  = 10
MAX_ELITIGATION_DOWNLOAD = 5

_rag = LegalFAISS()

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _prompt_user_confirmation(message: str) -> bool:
    print(f"\n[AGENT] {message}")
    choice = input("  Proceed? (y/n): ").strip().lower()
    return choice in ("y", "yes")

def _re_ingest():
    print("\n[AGENT] Re-ingesting newly downloaded cases...")
    import ingest
    ingest.main()

# ── RESEARCH PIPELINE ─────────────────────────────────────────────────────────

def _search_local(query: str, mode: str) -> dict:
    print(f"\n[AGENT] Step 1: Searching local database...")

    if mode == "case_summary":
        case_name = extract_case_name(query)
        if case_name:
            raw_chunks = _rag.search_by_source(query, case_name, top_k=12)
        else:
            raw_chunks = _rag.search(query, top_k=12)
    else:
        raw_chunks = _rag.search(query, top_k=12)

    chunks  = distil(query, raw_chunks, top_k=6)
    sources = list(dict.fromkeys(
        c.get("source") or c.get("meta", {}).get("source", "Unknown")
        for c in chunks
    ))

    print(f"  Found {len(chunks)} relevant chunks locally")

    return {
        "query":       query,
        "mode":        mode,
        "sources":     sources,
        "chunk_count": len(chunks),
        "raw_chunks":  chunks
    }

def _search_free_sources(query: str) -> list:
    saved = []
    try:
        from commonlii_tool import search_and_download as commonlii_search
        print(f"\n[AGENT] Step 2a: Searching CommonLII...")
        paths = commonlii_search(
            query,
            database="all",
            max_results=MAX_COMMONLII_RESULTS,
            max_download=MAX_COMMONLII_DOWNLOAD
        )
        saved.extend(paths)
    except Exception as e:
        print(f"  [SKIP] CommonLII unavailable: {e}")

    try:
        from elitigation_tool import search_and_download as judiciary_search
        print(f"\n[AGENT] Step 2b: Searching Judiciary.gov.sg recent judgments...")
        paths = judiciary_search(
            query,
            max_results=MAX_ELITIGATION_RESULTS,
            max_download=MAX_ELITIGATION_DOWNLOAD
        )
        saved.extend(paths)
    except Exception as e:
        print(f"  [SKIP] Judiciary.gov.sg unavailable: {e}")

    return saved

def _search_lawnet(query: str) -> list:
    try:
        from lawnet_tool import supervised_search
        print(f"\n[AGENT] Step 3: Falling back to Lawnet (supervised session)...")
        return supervised_search(query)
    except ImportError:
        print(f"  [SKIP] lawnet_tool.py not installed or playwright missing")
        return []
    except Exception as e:
        print(f"  [SKIP] Lawnet session failed: {e}")
        return []

# ── PUBLIC API ────────────────────────────────────────────────────────────────

def research(query: str, use_online: bool = True, use_lawnet: bool = False) -> str:
    """
    Full research pipeline:
    1. Search local vector store
    2. If insufficient — search CommonLII + judiciary.gov.sg (free)
    3. Re-ingest downloaded cases
    4. Re-search local store with new cases
    5. If still insufficient and use_lawnet=True — supervised Lawnet session
    6. Reason over all retrieved context
    7. Return formatted research memo
    """
    mode = route(query)
    print(f"\n[AGENT] Research mode: {mode.upper()}")
    print(f"[AGENT] Query: {query}")
    print("=" * 60)

    local_result = _search_local(query, mode)

    new_files = []
    if use_online and local_result["chunk_count"] < 3:
        print(f"\n[AGENT] Local results insufficient ({local_result['chunk_count']} chunks).")
        if _prompt_user_confirmation("Search CommonLII and judiciary.gov.sg for additional cases?"):
            new_files = _search_free_sources(query)
            if new_files:
                print(f"\n[AGENT] Downloaded {len(new_files)} new case(s). Re-ingesting...")
                _re_ingest()
                global _rag
                _rag = LegalFAISS()
                local_result = _search_local(query, mode)

    if use_lawnet and local_result["chunk_count"] < 2:
        print(f"\n[AGENT] Still insufficient results ({local_result['chunk_count']} chunks).")
        if _prompt_user_confirmation("Fall back to supervised Lawnet session?"):
            lawnet_files = _search_lawnet(query)
            if lawnet_files:
                _re_ingest()
                _rag = LegalFAISS()
                local_result = _search_local(query, mode)

    if local_result["chunk_count"] == 0:
        return format_no_results(query)

    print(f"\n[AGENT] Reasoning over {local_result['chunk_count']} chunks from "
          f"{len(local_result['sources'])} source(s)...")

    raw_answer = reason(
        query=query,
        context_chunks=local_result["raw_chunks"],
        mode=mode
    )

    response = format_response(
        raw=raw_answer,
        mode=mode,
        sources=local_result["sources"],
        query=query
    )

    if new_files:
        response += f"\n  [New cases downloaded and ingested: {len(new_files)}]\n"
        for f in new_files:
            response += f"    {os.path.basename(f)}\n"

    return response


def research_with_treatment(query: str, target_case: str, **kwargs) -> str:
    research_output  = research(query, **kwargs)
    treatment_report = analyse_treatment(target_case)
    treatment_output = format_treatment_report(treatment_report)
    return research_output + "\n" + treatment_output


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
            print("\nExiting.")
            break
        if not user_input:
            continue
        parts   = user_input.split(" ", 1)
        command = parts[0].lower()
        arg     = parts[1] if len(parts) > 1 else ""

        if command in ("exit", "quit", "q"):
            break
        elif command == "research" and arg:
            print(research(arg, use_online=True, use_lawnet=False))
        elif command == "local" and arg:
            print(research(arg, use_online=False, use_lawnet=False))
        elif command == "lawnet" and arg:
            print(research(arg, use_online=True, use_lawnet=True))
        elif command == "treatment" and arg:
            report = analyse_treatment(arg)
            print(format_treatment_report(report))
        else:
            print("  Unrecognised command. Type 'exit' to quit.")


if __name__ == "__main__":
    run_interactive()