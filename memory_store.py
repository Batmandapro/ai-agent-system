import json
import os
from datetime import datetime

class MemoryStore:
    def __init__(self, path="data/chats.json"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)

    def _load(self):
        with open(self.path, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def add_message(self, role, content, session_id="default"):
        data = self._load()

        data.append({
            "role": role,
            "content": content,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        self._save(data)

    def get_recent(self, limit=10, session_id="default"):
        data = self._load()

        filtered = [m for m in data if m["session_id"] == session_id]
        return filtered[-limit:]

    def get_all(self, session_id="default"):
        data = self._load()
        return [m for m in data if m["session_id"] == session_id]