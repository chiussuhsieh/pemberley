from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from retrieve import retrieve


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


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", END)   # for now the graph is just one node
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"query": "What does Darcy say about his own pride?"})
    print("query:", result["query"])
    print("retrieved", len(result["passages"]), "passages")
    for cite, text in zip(result["citations"], result["passages"]):
        print(f"  [{cite}] {text[:80]}...")