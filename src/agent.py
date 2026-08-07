import os
import sys
from typing import TypedDict, List

from dotenv import load_dotenv
from anthropic import Anthropic
from langgraph.graph import StateGraph, END

from retrieve import retrieve
from validate import validate_answer

load_dotenv()
client = Anthropic()

MODEL = "claude-haiku-4-5-20251001"
MAX_RETRIES = 2

# Character registry. Adding a character here makes them available to the router.
CHARACTERS = {
    "Elizabeth": {
        "name": "Elizabeth Bennet",
        "persona": "You are Elizabeth Bennet: quick-witted, playful, and sharply "
                   "observant, with a tendency toward irony and independent judgement.",
    },
    "Darcy": {
        "name": "Fitzwilliam Darcy",
        "persona": "You are Fitzwilliam Darcy: proud, reserved, and formal, with a "
                   "strong sense of honour and a guarded, deliberate manner of speaking.",
    },
}

DEFAULT_CHARACTER = "Elizabeth"


def route_character(query: str, current: str) -> str:
    """Decide who answers this turn.

    Explicit address wins: if the user names a known character (e.g. "Darcy, ..."),
    switch to them. Otherwise stay with the current character. Pure string logic,
    no LLM call.
    """
    lowered = query.lower()
    for key, info in CHARACTERS.items():
        # match either the short key ("darcy") or the first name ("fitzwilliam")
        first_name = info["name"].split()[0].lower()
        if lowered.startswith(key.lower()) or lowered.startswith(first_name):
            return key
    return current


class AgentState(TypedDict):
    query: str
    search_query: str
    character: str
    passages: List[str]
    citations: List[str]
    answer: str
    retry_count: int
    reason: str
    history: List[dict]


REWRITE_PROMPT = """Given the conversation history and a follow-up question,
rewrite the follow-up into a standalone question that can be understood without
the history. Resolve any pronouns or references (e.g. "her", "that", "his") to
the actual names or subjects from the history.

Only output the rewritten question, nothing else. If the question is already
standalone, output it unchanged.

Conversation history:
{history}

Follow-up question: {query}

Standalone question:"""


def rewrite_node(state: AgentState) -> AgentState:
    """Rewrite a context-dependent query into a standalone one for retrieval."""
    if not state["history"]:
        state["search_query"] = state["query"]
        return state

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in state["history"]
    )
    prompt = REWRITE_PROMPT.format(history=history_text, query=state["query"])
    msg = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    state["search_query"] = msg.content[0].text.strip()
    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Fetch relevant passages from the vector store, with their citations."""
    hits = retrieve(state["search_query"], k=5)
    state["passages"] = [h["text"] for h in hits]
    state["citations"] = [h["citation"] for h in hits]
    return state


SYSTEM_PROMPT = """{persona}

You are a character in Jane Austen's Pride and Prejudice. Stay fully in character:
speak in your own voice, your wit, your period.

You may ONLY use the reference passages below to answer. If the passages do not
support an answer, say so in character rather than inventing anything.

Every claim you make about events or opinions in the novel must be followed by a
citation in the exact form [Vol. X, Ch. Y], drawn from the passages provided.
{feedback}
Reference passages:
{context}
"""


def generate_node(state: AgentState) -> AgentState:
    """Generate an in-character reply grounded in the retrieved passages."""
    context = "\n\n".join(
        f"[{cite}] {text}"
        for cite, text in zip(state["citations"], state["passages"])
    )
    feedback = ""
    if state.get("reason"):
        feedback = f"\nYour previous answer was rejected: {state['reason']}. " \
                   f"Only cite chapters that appear in the passages below.\n"

    persona = CHARACTERS[state["character"]]["persona"]
    system = SYSTEM_PROMPT.format(
        persona=persona, context=context, feedback=feedback
    )
    messages = state["history"] + [{"role": "user", "content": state["query"]}]
    msg = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=system,
        messages=messages,
    )
    state["answer"] = msg.content[0].text
    return state


def validate_node(state: AgentState) -> AgentState:
    """Check the answer's citations. Record the failure reason if any."""
    passed, reason = validate_answer(state["answer"], state["citations"])
    state["reason"] = "" if passed else reason
    return state


def revise_node(state: AgentState) -> AgentState:
    """After retries are exhausted, replace the answer with an honest refusal."""
    state["answer"] = (
        "I am afraid I cannot find support for that in what I have before me. "
        "Perhaps you might ask me something else of the neighbourhood."
    )
    return state


def route_after_validate(state: AgentState) -> str:
    """Decide where to go after validation."""
    if not state["reason"]:
        return "pass"
    if state["retry_count"] < MAX_RETRIES:
        state["retry_count"] += 1
        return "retry"
    return "give_up"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("revise", revise_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"pass": END, "retry": "generate", "give_up": "revise"},
    )
    graph.add_edge("revise", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    history = []
    current_character = DEFAULT_CHARACTER
    names = ", ".join(info["name"] for info in CHARACTERS.values())
    print(f"You may speak with: {names}.")
    print("Address a character by name to switch (e.g. 'Darcy, ...'). Type 'quit' to end.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue
        # Decide who answers this turn (explicit address wins, else stay).
        current_character = route_character(query, current_character)
        name = CHARACTERS[current_character]["name"]
        result = app.invoke({
            "query": query,
            "search_query": "",
            "character": current_character,
            "history": history,
            "retry_count": 0,
            "reason": "",
        })
        answer = result["answer"]
        if result["search_query"] != query:
            print(f"[rewritten for search: {result['search_query']}]")
        print(f"\n{name}: {answer}\n")
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
