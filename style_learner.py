import os
import json
import re
from datetime import datetime
from llm import llm

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROFILE_DIR   = "data/profile"
SAMPLES_DIR   = "data/profile/samples"
ABOUT_ME_PATH = "data/profile/about-me.md"
RULES_PATH    = "data/profile/writing-rules.md"
MEMORY_PATH   = "data/profile/memory.md"

# ── SINGAPORE LEGAL WRITING STANDARDS (Layer 1 filter) ───────────────────────
# Based on Singapore court practice directions and established drafting principles
SG_LEGAL_STANDARDS = """
SINGAPORE LEGAL WRITING STANDARDS — QUALITY FILTER

Good legal writing in Singapore courts observes the following:

STRUCTURE
- Submissions are numbered sequentially (1, 2, 3...)
- Each paragraph contains one argument or proposition only
- IRAC or CREAC structure is used for legal arguments
- Headings are used to organise complex submissions
- Facts, issues, law, and application are clearly separated

LANGUAGE & TONE
- Formal but plain English — avoid archaic expressions (e.g. "hereinafter", "aforementioned")
- Active voice is preferred except in formal recitals
- Sentences are concise — avoid run-on sentences exceeding 3 clauses
- No colloquialisms, idioms, or informal expressions
- No unnecessary filler phrases (e.g. "it goes without saying", "needless to say")
- No padding or repetition of points already made
- Technical legal terms are used precisely and consistently

CITATIONS
- Cases cited as: Party v Party [Year] Volume Reporter Page (Court)
- Statutes cited with full short title and section number
- Paragraph numbers cited in square brackets e.g. at [23]
- Pinpoint citations are used — not just case names

ADVOCACY
- Arguments are made respectfully — "it is respectfully submitted"
- Concessions are made where appropriate — courts appreciate candour
- Strongest arguments lead; weaker ones follow or are omitted
- No hyperbole or overstatement of a case's strength
- Judicial observations are quoted accurately and in context

COMMON BAD HABITS TO FLAG
- Passive voice overuse (occasional passive in facts is fine; avoid in argument)
- Nominalisation (e.g. "make a determination" instead of "determine")
- Redundant pairs (e.g. "null and void", "cease and desist")
- Throat-clearing openings (e.g. "In this submission, we will...")
- Excessive hedging (e.g. "it may perhaps be argued that possibly...")
- Inconsistent tense within the same section
- Block quotations without analysis following them
"""

# ── FILE HELPERS ──────────────────────────────────────────────────────────────

def _ensure_dirs():
    for d in [PROFILE_DIR, SAMPLES_DIR]:
        os.makedirs(d, exist_ok=True)

def _read_file(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def _write_file(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def _append_file(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)

# ── TEXT EXTRACTION ───────────────────────────────────────────────────────────

def _extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF sample."""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(path)
        if text and len(text.strip()) > 100:
            return text.strip()
    except Exception as e:
        print(f"  [WARN] Could not extract {path}: {e}")
    return ""

def _load_samples() -> list:
    """Load all writing samples from data/profile/samples/."""
    samples = []

    # PDF samples
    for filename in os.listdir(SAMPLES_DIR):
        path = os.path.join(SAMPLES_DIR, filename)
        if filename.lower().endswith(".pdf"):
            print(f"  Reading: {filename}")
            text = _extract_text_from_pdf(path)
            if text:
                samples.append({"source": filename, "text": text})
        elif filename.lower().endswith(".txt"):
            print(f"  Reading: {filename}")
            text = _read_file(path)
            if text:
                samples.append({"source": filename, "text": text})

    return samples

# ── LAYER 1: QUALITY FILTER ───────────────────────────────────────────────────

def _filter_rule(rule: str) -> dict:
    """
    Run a single extracted style rule through the Singapore legal writing
    standards filter. Returns whether it is a good habit, bad habit, or neutral.
    """
    prompt = f"""You are a Singapore legal writing expert and judge of professional writing standards.

Below are the accepted Singapore court legal writing standards:
{SG_LEGAL_STANDARDS}

A writing pattern has been extracted from a lawyer's past work:
"{rule}"

Classify this pattern as one of:
- GOOD HABIT: Consistent with Singapore legal writing standards — keep
- BAD HABIT: Inconsistent with standards, or a known weakness in legal writing — flag for review
- NEUTRAL: Neither good nor bad — stylistic preference only

Respond in exactly this format:
Classification: [GOOD HABIT / BAD HABIT / NEUTRAL]
Reason: [one sentence explanation]
"""
    response = llm.invoke(prompt).content.strip()

    classification = "NEUTRAL"
    reason         = ""

    for line in response.splitlines():
        if line.startswith("Classification:"):
            classification = line.replace("Classification:", "").strip()
        elif line.startswith("Reason:"):
            reason = line.replace("Reason:", "").strip()

    return {
        "rule":           rule,
        "classification": classification,
        "reason":         reason
    }

# ── LAYER 2: USER APPROVAL ────────────────────────────────────────────────────

def _present_for_approval(filtered_rules: list) -> list:
    """
    Present extracted rules to the user for approval.
    Returns only the approved rules.
    """
    approved = []

    print("\n" + "=" * 60)
    print("  WRITING STYLE REVIEW")
    print("  Review each extracted pattern.")
    print("  Y = keep  |  N = discard  |  E = edit before keeping")
    print("=" * 60)

    for i, item in enumerate(filtered_rules, 1):
        rule           = item["rule"]
        classification = item["classification"]
        reason         = item["reason"]

        # Flag bad habits clearly
        flag = ""
        if "BAD" in classification.upper():
            flag = "  ⚠ POTENTIAL BAD HABIT"

        print(f"\n{i}. {rule}{flag}")
        print(f"   [{classification}] {reason}")

        while True:
            choice = input("   Keep? (Y/N/E): ").strip().upper()
            if choice == "Y":
                approved.append(rule)
                break
            elif choice == "N":
                print("   Discarded.")
                break
            elif choice == "E":
                edited = input("   Edit to: ").strip()
                if edited:
                    approved.append(edited)
                    print("   Saved (edited).")
                break
            else:
                print("   Please enter Y, N, or E.")

    return approved

# ── STYLE EXTRACTION ──────────────────────────────────────────────────────────

def _extract_style_rules(samples: list) -> list:
    """
    Use the LLM to extract writing style patterns from samples.
    Returns a raw list of pattern strings.
    """
    combined = ""
    for s in samples[:3]:   # limit to first 3 samples to avoid token overflow
        combined += f"\n\n--- From: {s['source']} ---\n{s['text'][:3000]}"

    prompt = f"""You are a legal writing analyst.

Analyse the following excerpts from a Singapore criminal lawyer's written work.
Extract exactly 10–15 specific, observable writing patterns and habits.

Be precise and concrete — not generic advice. For example:
- "Opens mitigation pleas with the client's personal background before the offence facts"
- "Uses 'it is respectfully submitted' before every substantive legal argument"
- "Numbers each paragraph sequentially throughout the entire submission"
- "Cites cases as Party v Party [Year] SLR Page without always including the court level"

Avoid generic observations like "writes clearly" or "uses formal language".

WRITING SAMPLES:
{combined}

List each pattern on its own line, numbered 1 to 15.
Do not include any other commentary.
"""

    response = llm.invoke(prompt).content.strip()

    # Parse numbered list
    rules = []
    for line in response.splitlines():
        line = line.strip()
        match = re.match(r'^\d+[\.\)]\s+(.+)', line)
        if match:
            rules.append(match.group(1).strip())

    return rules

# ── PROFILE INITIALISATION ────────────────────────────────────────────────────

def _create_about_me():
    """Interactive setup to create about-me.md."""
    print("\n" + "=" * 60)
    print("  ABOUT ME SETUP")
    print("  Answer the following questions to build your profile.")
    print("=" * 60)

    questions = [
        ("Your name",                          "name"),
        ("Your role / title",                  "role"),
        ("Your firm or chambers",              "firm"),
        ("Your primary practice area",         "practice_area"),
        ("Courts you appear in most often",    "courts"),
        ("Types of clients you typically act for", "clients"),
        ("Any current active matters or focus areas", "current_focus"),
    ]

    answers = {}
    for question, key in questions:
        answer = input(f"\n  {question}:\n  > ").strip()
        answers[key] = answer

    content = f"""# About Me
*Last updated: {datetime.now().strftime('%d %b %Y')}*

## Personal Details
- **Name:** {answers['name']}
- **Role:** {answers['role']}
- **Firm / Chambers:** {answers['firm']}

## Practice
- **Primary area:** {answers['practice_area']}
- **Courts:** {answers['courts']}
- **Typical clients:** {answers['clients']}

## Current Focus
{answers['current_focus']}
"""
    _write_file(ABOUT_ME_PATH, content)
    print(f"\n  [SAVED] {ABOUT_ME_PATH}")
    return content

def _create_writing_rules(approved_rules: list):
    """Save approved writing rules to writing-rules.md."""
    rules_text = "\n".join(f"- {r}" for r in approved_rules)

    content = f"""# Writing Rules
*Last updated: {datetime.now().strftime('%d %b %Y')}*
*Rules approved and verified against Singapore legal writing standards.*

## Style Patterns (Approved)
{rules_text}

## What to Avoid
- Never use AI-sounding filler phrases ("It is important to note that...", "Certainly", "Absolutely")
- Never use nominalisation where a verb will do ("make a determination" → "determine")
- Never pad arguments — every sentence must advance the submission
- Never use block quotes without analytical commentary immediately following
- Never overstate — if a case does not directly support the point, say so
- Never use inconsistent citation formats within the same document

## Tone
- Formal, precise, and measured
- Respectful to the court — never adversarial in tone toward the judge
- Candid — acknowledge weaknesses in the case where appropriate
- Confident in argument but never hyperbolic
"""
    _write_file(RULES_PATH, content)
    print(f"  [SAVED] {RULES_PATH}")

def _create_memory():
    """Initialise an empty memory.md."""
    content = f"""# Memory Log
*Initialised: {datetime.now().strftime('%d %b %Y')}*
*This file is updated automatically after drafting sessions and manually via add_manual_note().*

## Judicial Observations
*(Notes on specific judges — formatting preferences, receptiveness to arguments, etc.)*


## Arguments & Outcomes
*(Arguments that succeeded or failed in court — to inform future drafting.)*


## Case Notes
*(Ongoing matters and key developments.)*


## General Preferences
*(Any other preferences or observations logged over time.)*

"""
    _write_file(MEMORY_PATH, content)
    print(f"  [SAVED] {MEMORY_PATH}")

# ── PUBLIC API ────────────────────────────────────────────────────────────────

def bootstrap(paste_sample: str = None):
    """
    One-time setup. Call this once to initialise your profile.

    Workflow:
    1. Creates about-me.md via interactive Q&A
    2. Loads PDF/TXT samples from data/profile/samples/
    3. Optionally accepts a pasted writing sample
    4. Extracts style rules via LLM
    5. Filters each rule against SG legal writing standards
    6. Presents rules for your approval
    7. Saves approved rules to writing-rules.md
    8. Initialises memory.md

    Args:
        paste_sample: Optional string — paste a writing excerpt directly
    """
    _ensure_dirs()

    print("\n" + "=" * 60)
    print("  LEGAL AI — STYLE LEARNING BOOTSTRAP")
    print("=" * 60)

    # Step 1 — About Me
    _create_about_me()

    # Step 2 — Load samples
    print(f"\n[STYLE] Loading writing samples from {SAMPLES_DIR}...")
    samples = _load_samples()

    if paste_sample:
        samples.append({"source": "pasted_sample", "text": paste_sample})
        print(f"  Added pasted sample ({len(paste_sample)} chars)")

    if not samples:
        print(f"\n  [WARN] No samples found in {SAMPLES_DIR}")
        print(f"  Drop PDF or TXT files into {SAMPLES_DIR} and re-run bootstrap()")
        print(f"  Creating writing-rules.md with default rules only...")
        _create_writing_rules([])
        _create_memory()
        return

    print(f"  Loaded {len(samples)} sample(s)")

    # Step 3 — Extract patterns
    print(f"\n[STYLE] Analysing writing patterns (this may take a moment)...")
    raw_rules = _extract_style_rules(samples)
    print(f"  Extracted {len(raw_rules)} patterns")

    # Step 4 — Filter against SG standards (Layer 1)
    print(f"\n[STYLE] Checking patterns against Singapore legal writing standards...")
    filtered = []
    for i, rule in enumerate(raw_rules):
        print(f"  Checking {i+1}/{len(raw_rules)}...", end="\r")
        filtered.append(_filter_rule(rule))
    print(f"  Quality check complete.                    ")

    # Step 5 — User approval (Layer 2)
    approved = _present_for_approval(filtered)
    print(f"\n  {len(approved)} rule(s) approved out of {len(raw_rules)} extracted.")

    # Step 6 — Save
    _create_writing_rules(approved)
    _create_memory()

    print("\n" + "=" * 60)
    print("  Bootstrap complete.")
    print(f"  Profile saved to {PROFILE_DIR}/")
    print("  The LLM will now read your profile before every drafting task.")
    print("=" * 60)


def apply_style(prompt: str, mode: str) -> str:
    """
    Inject your writing profile into a drafting prompt.
    Called automatically by reasoning_engine.py for 'drafting' mode.

    Args:
        prompt: The base prompt for the LLM
        mode:   Intent mode — only injects style for 'drafting' mode

    Returns:
        Prompt with profile injected, or original prompt unchanged
    """
    if mode != "drafting":
        return prompt

    about_me     = _read_file(ABOUT_ME_PATH)
    writing_rules = _read_file(RULES_PATH)
    memory       = _read_file(MEMORY_PATH)

    if not any([about_me, writing_rules, memory]):
        return prompt   # Profile not yet set up — return unchanged

    injection = f"""
LAWYER PROFILE (read before drafting):
{about_me}

WRITING RULES (follow precisely):
{writing_rules}

MEMORY & CONTEXT:
{memory}

Apply the above profile to everything you draft. Write as this lawyer would write —
using their style, their structure, their tone. Do not sound like an AI.
Do not use phrases like "It is important to note", "Certainly", "Absolutely",
"I'd be happy to", or any AI-sounding filler.

"""
    return injection + prompt


def update_memory(session_notes: str, category: str = "General Preferences"):
    """
    Append a new observation to memory.md after a drafting session.
    Called automatically after drafting, or manually at any time.

    Args:
        session_notes: What to log
        category:      Which section to append to
                       Options: "Judicial Observations", "Arguments & Outcomes",
                                "Case Notes", "General Preferences"
    """
    timestamp = datetime.now().strftime("%d %b %Y, %H:%M")
    entry     = f"\n- [{timestamp}] {session_notes}"

    memory = _read_file(MEMORY_PATH)

    if category in memory:
        # Append under the correct section
        memory = memory.replace(
            f"## {category}",
            f"## {category}{entry}"
        )
        _write_file(MEMORY_PATH, memory)
    else:
        # Append at end if section not found
        _append_file(MEMORY_PATH, f"\n## {category}\n{entry}\n")

    print(f"[MEMORY] Logged to '{category}'")


def add_manual_note(note: str, category: str = "General Preferences"):
    """
    Manually add a note to memory.md at any time.

    Example:
        add_manual_note("Judge X prefers numbered submissions and dislikes block quotes",
                        "Judicial Observations")

    Args:
        note:     The note to add
        category: Which memory section to add it to
    """
    update_memory(note, category)
    print(f"[MEMORY] Note added: {note[:80]}...")


def review_rules():
    """Print all current writing rules for review."""
    rules = _read_file(RULES_PATH)
    if not rules:
        print("[STYLE] No writing rules found. Run bootstrap() first.")
        return
    print("\n" + "=" * 60)
    print("  CURRENT WRITING RULES")
    print("=" * 60)
    print(rules)


def review_memory():
    """Print the full memory log."""
    memory = _read_file(MEMORY_PATH)
    if not memory:
        print("[MEMORY] Memory log is empty. Run bootstrap() first.")
        return
    print("\n" + "=" * 60)
    print("  MEMORY LOG")
    print("=" * 60)
    print(memory)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("""
Style Learner — Commands:
  python style_learner.py bootstrap          — First-time setup
  python style_learner.py review-rules       — View current writing rules
  python style_learner.py review-memory      — View memory log
  python style_learner.py note "your note"   — Add a manual memory note
  python style_learner.py note "your note" "Judicial Observations"
        """)

    elif sys.argv[1] == "bootstrap":
        bootstrap()

    elif sys.argv[1] == "review-rules":
        review_rules()

    elif sys.argv[1] == "review-memory":
        review_memory()

    elif sys.argv[1] == "note" and len(sys.argv) >= 3:
        note     = sys.argv[2]
        category = sys.argv[3] if len(sys.argv) >= 4 else "General Preferences"
        add_manual_note(note, category)

    else:
        print(f"Unknown command: {sys.argv[1]}")