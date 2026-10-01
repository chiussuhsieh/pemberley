from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import build_graph, CHARACTERS, DEFAULT_CHARACTER, route_character
import session

app = FastAPI(title="Pemberley")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()


# The client now holds only a session id. Conversation history and the current
# character live in Redis, keyed by that id, so a conversation survives a page
# reload or a return visit.
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    character: str
    character_name: str
    search_query: str
    session_id: str


@app.get("/characters")
def characters():
    """List the characters a client can address."""
    return {k: v["name"] for k, v in CHARACTERS.items()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Run one conversation turn, loading and saving history from Redis."""
    # New conversation if no session id was supplied.
    session_id = req.session_id or session.new_session()

    state = session.load(session_id)
    history = state["history"]
    current = state["character"]

    # Decide who answers (explicit address wins, else stay with current).
    character = route_character(req.query, current)

    result = graph.invoke({
        "query": req.query,
        "search_query": "",
        "character": character,
        "history": history,
        "retry_count": 0,
        "reason": "",
    })
    answer = result["answer"]

    # Append this turn and persist back to Redis.
    history = history + [
        {"role": "user", "content": req.query},
        {"role": "assistant", "content": answer},
    ]
    session.save(session_id, history, character)

    return ChatResponse(
        answer=answer,
        citations=result["citations"],
        character=character,
        character_name=CHARACTERS[character]["name"],
        search_query=result["search_query"],
        session_id=session_id,
    )
