# app/memory.py
from typing import List, Dict

MAX_TURNS = 8  # keep last 8 dialog turns


class MemoryStore:
    def __init__(self):
        # { user_id: { "history": [...], "summary": "" } }
        self.sessions: Dict[str, Dict] = {}

    def init(self, user_id: str):
        if user_id not in self.sessions:
            self.sessions[user_id] = {"history": [], "summary": ""}

    def add_turn(self, user_id: str, role: str, content: str):
        """Append a new turn and trim if necessary."""
        self.init(user_id)

        history = self.sessions[user_id]["history"]
        history.append({"role": role, "content": content})

        # Trim memory
        if len(history) > MAX_TURNS:
            # Remove oldest two messages (1 user + 1 assistant)
            removed = history[:-MAX_TURNS]
            self.sessions[user_id]["history"] = history[-MAX_TURNS:]

            # Optional future: summarize removed portion
            # summary_text = summarize_long_history(removed)
            # self.sessions[user_id]["summary"] += summary_text

    def get_history(self, user_id: str) -> List[Dict]:
        self.init(user_id)
        return self.sessions[user_id]["history"]

    def get_summary(self, user_id: str) -> str:
        self.init(user_id)
        return self.sessions[user_id]["summary"]


memory_store = MemoryStore()
