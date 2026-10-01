import json
import uuid

import redis

# One shared connection. decode_responses=True so we get str back, not bytes.
_r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Conversations expire after this many seconds of inactivity, so Redis doesn't
# grow without bound. Any read/write refreshes the clock.
TTL_SECONDS = 7 * 24 * 60 * 60   # 7 days


def new_session() -> str:
    """Create a fresh session id for a new conversation."""
    return uuid.uuid4().hex


def _key(session_id: str) -> str:
    return f"pemberley:session:{session_id}"


def load(session_id: str) -> dict:
    """Return {history, character} for a session, or defaults if unknown."""
    raw = _r.get(_key(session_id))
    if raw is None:
        return {"history": [], "character": "Elizabeth"}
    return json.loads(raw)


def save(session_id: str, history: list, character: str) -> None:
    """Persist a session's state and refresh its expiry."""
    data = json.dumps({"history": history, "character": character})
    _r.set(_key(session_id), data, ex=TTL_SECONDS)
