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
CHARACTER = "Elizabeth Bennet"
MAX_RETRIES = 2


class AgentState(TypedDict):
    query: str
    search_query: str
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


SYSTEM_PROMPT = """You are {character} from Jane Austen's Pride and Prejudice.
Stay fully in character: speak in her voice, her wit, her period.

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

    system = SYSTEM_PROMPT.format(
        character=CHARACTER, context=context, feedback=feedback
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
    print(f"You are speaking with {CHARACTER}. Type 'quit' to end.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue
        result = app.invoke({
            "query": query,
            "search_query": "",
            "history": history,
            "retry_count": 0,
            "reason": "",
        })
        answer = result["answer"]
        if result["search_query"] != query:
            print(f"[rewritten for search: {result['search_query']}]")
        print(f"\n{CHARACTER}: {answer}\n")
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
