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


# AgentState is the shared "briefcase" passed between nodes:
#   query        user's question           (written by caller)
#   passages     retrieved source text     (written by retrieve, read by generate)
#   citations    Vol./Ch. references       (written by retrieve, read by validate)
#   answer       the character's reply      (written by generate, read by validate)
#   retry_count  how many times we retried  (managed by validate routing)
#   reason       why validation failed      (written by validate, read by generate)
#   history      prior turns in this session ([{"role", "content"}, ...])
class AgentState(TypedDict):
    query: str
    passages: List[str]
    citations: List[str]
    answer: str
    retry_count: int
    reason: str
    history: List[dict]


def retrieve_node(state: AgentState) -> AgentState:
    """Fetch relevant passages from the vector store, with their citations."""
    hits = retrieve(state["query"], k=5)
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
    # On a retry, tell the model why the previous attempt was rejected.
    feedback = ""
    if state.get("reason"):
        feedback = f"\nYour previous answer was rejected: {state['reason']}. " \
                   f"Only cite chapters that appear in the passages below.\n"

    system = SYSTEM_PROMPT.format(
        character=CHARACTER, context=context, feedback=feedback
    )
    # Prepend prior turns so the character remembers the conversation.
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
        return "pass"                      # validation succeeded
    if state["retry_count"] < MAX_RETRIES:
        state["retry_count"] += 1
        return "retry"                     # try generating again
    return "give_up"                       # exhausted retries


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("revise", revise_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")

    # Conditional branch after validate.
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
            "history": history,
            "retry_count": 0,
            "reason": "",
        })
        answer = result["answer"]
        print(f"\n{CHARACTER}: {answer}\n")
        # Append this turn to history so the next turn remembers it.
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})