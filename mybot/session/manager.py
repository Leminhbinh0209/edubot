import json
from pathlib import Path
from datetime import datetime
from typing import Any

class Session:
    def __init__(self, key: str):
        self.key = key
        self.messages: list[dict[str, Any]] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in recent
        ]
class SessionManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.sessions_dir = data_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}
    def get_session_path(self, key: str) -> Path:
        safe_key = key.replace(":", "_").replace("/", "_")
        return self.sessions_dir / f"{safe_key}.json"
    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]
        path = self.get_session_path(key)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                session = Session(key=key)
                session.messages = data.get("messages", [])
                if data.get("created_at"):
                    session.created_at = datetime.fromisoformat(data["created_at"])
                self._cache[key] = session
                return session
            except Exception:
                pass
        session = Session(key=key)
        self._cache[key] = session
        return session
    def save(self, session: Session) -> None:
        path = self.get_session_path(session.key)
        data = {
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "messages": session.messages
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._cache[session.key] = session