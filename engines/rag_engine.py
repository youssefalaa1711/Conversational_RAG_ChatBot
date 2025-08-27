# engines/rag_engine.py
from typing import List, Tuple
from langchain.memory import ChatMessageHistory

class RagEngine:
    def __init__(self, chain, history_store: dict[str, ChatMessageHistory]):
        self.chain = chain
        self.history_store = history_store

    def answer(self, session_id: str, question: str) -> str:
        # Call the chain with session-aware config
        resp = self.chain.invoke(
            {"question": question},
            config={"session_id": session_id},   # pass clean session_id
        )
        return resp

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Return chat history as [(role, message)] tuples for UI display."""
        h = self.history_store.get(session_id)
        if not h:
            return []
        out = []
        for m in h.messages:
            role = "user" if m.type == "human" else "assistant"
            out.append((role, m.content))
        return out
