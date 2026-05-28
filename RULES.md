# Legal AI System — Assistant Rules

**Version:** v1.0
**Last Updated:** 28 May 2026

---

## 1. Purpose

This file is read by the AI assistant at the start of every working session. It governs how the assistant behaves across all interactions relating to the Singapore Legal AI project. Its purpose is to ensure consistency, prevent regressions, and maintain a high standard of code quality and communication throughout development. All instructions herein take precedence over default assistant behaviour unless explicitly overridden by the user in-session.

---

## 2. Core Behaviour Rules

1. **Think before responding.** Take time to reason through the proper solution before writing any code or advice. Do not rush. Do not over-complicate simple problems — identify the simplest correct solution first.

2. **All code must be complete and in full.** Never ask the user to replace a specific portion, patch a section, or fill in the rest themselves. Every file provided must be the entire file, ready to save and use as-is.

3. **Verify coherence across the project before every update.** Before proposing any change or improvement, check that it is consistent with all other files in the project. Never use filler values, generic variable names, or placeholder comments that could leave the codebase in a broken or ambiguous state.

4. **Be upfront and direct about risks.** If something will not work, or if a proposed approach has known risks or limitations, say so plainly and clearly before writing any code. Do not bury caveats after the solution.

5. **Assess full pipeline impact before every change.** Before making any modification, state which other files are affected and exactly what must change in them. If no other files are affected, say so explicitly.

6. **Never use placeholder values.** Do not use values such as `"your_model_here"`, `TODO`, `<insert_key>`, or any similar stand-in. Use only actual values confirmed from the project's source files.

7. **Wrap all code in clear delimiters.** Every code block must be enclosed as follows:

   ```
   === CODE BEGIN: <filename> ===
   
   <full file contents>
   
   === CODE END: <filename> ===
   ```

8. **Use British spelling throughout.** All code comments, docstrings, variable names where readable text is used, and output text must follow British English spelling conventions (e.g. *analyse*, *colour*, *initialise*, *serialise*, *behaviour*).

---

## 3. End-of-Session Review

At the end of each working session — triggered when the user signals they are done for the day (e.g. by saying **"I'm going to sleep"**, **"end of session review"**, or similar) — the assistant must conduct a structured review of all code changes made during that session.

The review must cover the following three areas:

- **(a) Regressions and inconsistencies** — any changes made during the session that may have introduced bugs, broken existing functionality, or created inconsistencies with other parts of the codebase.
- **(b) Deferred improvements** — improvements that were identified or discussed but deliberately set aside for a future session. These should be noted clearly so they are not forgotten.
- **(c) Unresolved flags** — anything that was flagged as a concern, uncertainty, or risk during the session but was not fully resolved before signing off.

The review must be output as a **plain text summary** with a dated heading in the following format:

```
END-OF-SESSION REVIEW — 28 May 2026

(a) Regressions / Inconsistencies
...

(b) Deferred Improvements
...

(c) Unresolved Flags
...
```

This review should be offered proactively whenever the user signals the end of a session. It does not require a specific command — natural language such as *"I'm going to sleep"* or *"that's all for today"* is sufficient to trigger it.

---

## 4. Prompt Engineering Best Practices

These principles are drawn from the Claude CoWork methodology (BetterCreating / Simon) and apply to how this assistant is configured and used.

- **Global instructions are the single most important setup step.** They are read at the start of every session and shape all subsequent behaviour. Keep them current, specific, and comprehensive.

- **Be specific rather than vague.** Vague instructions produce vague, inconsistent results. Every rule in this file should be precise enough that there is only one reasonable interpretation.

- **Use a layered instruction system.** The full context stack is:
  1. Global rules (this file)
  2. Per-project context (project-specific rules below)
  3. Memory (updated at the end of each session)
  4. Writing style and tone preferences

- **Ask rather than assume.** If it is unclear whether a new request represents a fresh approach or a continuation of existing work, ask the user to clarify before proceeding. Do not make assumptions that could conflict with prior decisions.

- **Challenge the user when there is a better way.** If the user proposes an approach that has a clearly superior alternative, say so directly and explain why. This is a collaborative process — honest pushback is valued over compliance.

- **Never delete, send, or publish anything without explicit user confirmation.** This applies to files, database entries, API calls, GitHub commits, and any other irreversible or externally visible actions.

- **Update memory at the end of each session.** Session context, key decisions, and the current state of the project should be preserved so that the next session can resume without loss of continuity.

- **Reference all supporting context files explicitly.** Any file that provides important project context (e.g. schemas, configuration files, data models) should be referenced explicitly in the global instructions so that it is always loaded and considered.

---

## 5. Project-Specific Rules

These rules apply specifically to the Singapore Legal AI system and override general defaults where they conflict.

| Setting | Value |
|---|---|
| **GitHub branch** | `master` (never `main`) |
| **Vector DB key for text** | `"text"` (not `"chunk"` — this was a historical bug; never revert) |
| **Flask API port** | `5000` |
| **Ollama base URL** | `http://localhost:11434` |
| **LLM model** | `llama3.1` |
| **Embedding model** | `nomic-embed-text` |
| **Vector store path** | `data/cases_db.json` |
| **Python venv path** | `C:\Users\Admin\Desktop\ai-agent-system\venv\Scripts\python.exe` |

### Atomic File Saves

All file write operations must be **atomic**. This means:

1. Write the new content to a temporary file with a `.tmp` extension.
2. Use `os.replace()` to move the `.tmp` file to the target path in a single operation.

This prevents data corruption if the process is interrupted mid-write. Never write directly to the target file in a single open-and-write call for any file that holds persistent state.

---

## 6. Scheduled Tasks — Honest Note

True background scheduled tasks — such as a nightly review that runs automatically while the computer is off or the application is closed — are **not natively possible** in a local Python setup without additional tooling outside the application itself.

### Options for Windows

- **Windows Task Scheduler** (recommended, free, built into Windows): Can be configured to run a Python script at a set time, even on a schedule. This is the most practical option for a local Windows machine and requires no additional software.
- **A cron daemon** (Linux/macOS equivalent): Not applicable on Windows without WSL or a third-party tool.

### Within an Active Session

End-of-session reviews and other structured outputs can always be triggered manually during an active session. The user does not need a scheduled task for these — simply signal the end of the session using natural language (see Section 3 above).

---

## 7. How to Update This File

1. Open `RULES.md` in **VS Code** (or any text editor).
2. Navigate to the relevant section and make your edits.
3. Save the file.
4. Commit and push to **GitHub `master`** so that the updated rules persist across machines and sessions:

   ```bash
   git add RULES.md
   git commit -m "Update RULES.md: <brief description of change>"
   git push origin master
   ```

Each section is self-contained, so edits to one section should not require changes elsewhere in this file unless version history needs updating.

---

## 8. Version History

| Version | Date | Change |
|---|---|---|
| v1.0 | 28 May 2026 | Initial rules file |
