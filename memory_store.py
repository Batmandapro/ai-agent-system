# FILE: memory_store.py
# LOCATION: C:\Users\Admin\Desktop\ai-agent-system\memory_store.py
# ACTION: Replace entire file

import json
import os
from datetime import datetime


class MemoryStore:
    """Persistent chat memory store backed by a JSON file.

    Improvements over the original:
    1. Atomic saves — writes to a .tmp file then uses os.replace() to prevent
       corruption if the process is interrupted during a write.
    2. Explicit encoding="utf-8" on all file operations to handle Unicode
       content reliably on Windows.
    3. Safe load — catches JSONDecodeError and OSError on _load() so a
       corrupted file does not crash the session; returns an empty list instead.
    4. os.makedirs uses exist_ok=True so the data directory is created once
       at construction without raising an error if it already exists.
    """

    def __init__(self, path="data/chats.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if not os.path.exists(self.path):
            self._atomic_save([])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self):
        """Load and return the full message list from disk.

        Returns an empty list if the file is missing, empty, or corrupt.
        """
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, ValueError):
            return []

    def _atomic_save(self, data):
        """Write data to disk atomically.

        Writes to <path>.tmp first, then calls os.replace() to swap it in.
        This ensures the file is never left in a partially-written state.
        """
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_message(self, role, content, session_id="default"):
        """Append a message to the store and persist atomically."""
        data = self._load()

        data.append({
            "role":       role,
            "content":    content,
            "session_id": session_id,
            "timestamp":  datetime.utcnow().isoformat(),
        })

        self._atomic_save(data)

    def get_recent(self, limit=10, session_id="default"):
        """Return the most recent *limit* messages for a given session."""
        data = self._load()
        filtered = [m for m in data if m["session_id"] == session_id]
        return filtered[-limit:]

    def get_all(self, session_id="default"):
        """Return all messages for a given session."""
        data = self._load()
        return [m for m in data if m["session_id"] == session_id]
