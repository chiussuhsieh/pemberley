import json
import os
import uuid

import redis

# Connection URL comes from the environment in deployment (Render sets REDIS_URL);
# falls back to local Docker Redis for development.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

TTL_SECONDS = 7 * 24 * 60 * 60   # 7 days


def new_session() -> str:
    return uuid.uuid4().hex


def _key(session_id: str) -> str:
    return f"pemberley:session:{session_id}"


def load(session_id: str) -> dict:
    raw = _r.get(_key(session_id))
    if raw is None:
        return {"history": [], "character": "Elizabeth"}
    return json.loads(raw)


def save(session_id: str, history: list, character: str) -> None:
    data = json.dumps({"history": history, "character": character})
    _r.set(_key(session_id), data, ex=TTL_SECONDS)
