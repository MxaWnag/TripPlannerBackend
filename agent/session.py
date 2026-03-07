"""
In-memory session store for multi-turn chat. Keyed by session_id; each session holds (user, assistant) message pairs.
For production, replace with Redis or DB.
"""
import uuid
from typing import Any, List, Optional

# session_id -> list of {"role": "user" | "assistant", "content": str}
_SESSIONS: dict[str, List[dict[str, Any]]] = {}

# Limit history to last N messages (N/2 turns) to avoid unbounded context
MAX_HISTORY_MESSAGES = int(__import__("os").environ.get("AGENT_SESSION_MAX_MESSAGES", "20"))


def create_session_id() -> str:
    return str(uuid.uuid4())


def get_history(session_id: str) -> List[dict[str, Any]]:
    """Return list of {role, content}; empty if unknown session."""
    return list(_SESSIONS.get(session_id, []))


def append_turn(session_id: str, user_content: str, assistant_content: str) -> None:
    """Append one (user, assistant) turn and trim to MAX_HISTORY_MESSAGES."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = []
    _SESSIONS[session_id].append({"role": "user", "content": user_content})
    _SESSIONS[session_id].append({"role": "assistant", "content": assistant_content})
    # Keep last N messages
    if len(_SESSIONS[session_id]) > MAX_HISTORY_MESSAGES:
        _SESSIONS[session_id] = _SESSIONS[session_id][-MAX_HISTORY_MESSAGES:]


def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Return existing session_id or create a new one."""
    if session_id and session_id.strip():
        return session_id.strip()
    return create_session_id()
