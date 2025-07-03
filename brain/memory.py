# brain/memory.py
class ChatMemory:
    """Simple in‑memory conversation + mood logger."""
    def __init__(self):
        self.history = []      # list[(role, text)]
        self.moodlog = []      # list[(timestamp, valence, arousal)]

    # -- interface ----------------------------------------------------------
    def add(self, role: str, text: str):
        self.history.append((role, text))

    def last_is_user(self):
        return self.history and self.history[-1][0] == "user"
