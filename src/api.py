from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from agent import build_graph, CHARACTERS, DEFAULT_CHARACTER, route_character

app = FastAPI(title="Pemberley")
graph = build_graph()


# Request/response schemas. The client holds conversation state and sends it
# with each request, so the API stays stateless (any instance can serve any turn).
class ChatRequest(BaseModel):
    query: str
    history: List[dict] = []
    character: str = DEFAULT_CHARACTER


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    character: str
    character_name: str
    search_query: str


@app.get("/characters")
def characters():
    """List the characters a client can address."""
    return {k: v["name"] for k, v in CHARACTERS.items()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Run one conversation turn and return the character's grounded reply."""
    # Decide who answers (explicit address wins, else stay with current).
    character = route_character(req.query, req.character)

    result = graph.invoke({
        "query": req.query,
        "search_query": "",
        "character": character,
        "history": req.history,
        "retry_count": 0,
        "reason": "",
    })

    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"],
        character=character,
        character_name=CHARACTERS[character]["name"],
        search_query=result["search_query"],
    )
