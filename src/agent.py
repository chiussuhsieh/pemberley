import os
from typing import TypedDict, List
from dotenv import load_dotenv
from anthropic import Anthropic
from langgraph.graph import StateGraph, END

from retrieve import retrieve

load_dotenv()
client = Anthropic()

MODEL = "claude-haiku-4-5-20251001"   # cheap model for development

# AgentState is the shared "briefcase" passed between nodes. Each field is written
# by one node and read by a later one:
#   query      user's question           (written by caller, read by retrieve)
#   passages   retrieved source text     (written by retrieve, read by generate)
#   citations  Vol./Ch. references       (written by retrieve, read by validate)
#   answer     the character's reply      (written by generate, read by validate)
class AgentState(TypedDict):
    query: str
    passages: List[str]
    citations: List[str]
    answer: str


def retrieve_node(state: AgentState) -> AgentState:
    """Fetch relevant passages from the vector store, with their citations."""
    hits = retrieve(state["query"], k=5)
    state["passages"] = [h["text"] for h in hits]
    state["citations"] = [h["citation"] for h in hits]
    return state

CHARACTER = "Elizabeth Bennet"

SYSTEM_PROMPT = """You are {character} from Jane Austen's Pride and Prejudice.
Stay fully in character: speak in her voice, her wit, her period.

You may ONLY use the reference passages below to answer. If the passages do not
support an answer, say so in character rather than inventing anything.

Every claim you make about events or opinions in the novel must be followed by a
citation in the exact form [Vol. X, Ch. Y], drawn from the passages provided.

Reference passages:
{context}
"""


def generate_node(state: AgentState) -> AgentState:
    """Generate an in-character reply grounded in the retrieved passages."""
    context = "\n\n".join(
        f"[{cite}] {text}"
        for cite, text in zip(state["citations"], state["passages"])
    )
    system = SYSTEM_PROMPT.format(character=CHARACTER, context=context)

    msg = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": state["query"]}],
    )
    state["answer"] = msg.content[0].text
    return state

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")   # retrieve then generate
    graph.add_edge("generate", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"query": "What does Darcy say about his own pride?"})
    print("Q:", result["query"])
    print()
    print(f"{CHARACTER}:")
    print(result["answer"])
    print()
    print("retrieved citations:", result["citations"])