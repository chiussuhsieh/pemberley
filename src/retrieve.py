import sys

import chromadb

from config import COLLECTION, CHROMA_PATH, get_embedding_function


def retrieve(query, k=5):
    col = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(
        name=COLLECTION,
        embedding_function=get_embedding_function(),
    )
    res = col.query(query_texts=[query], n_results=k)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"text": doc, "citation": meta["citation"], "distance": dist})
    return hits


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What does Darcy say about his own pride?"
    print(f"query: {query}\n")
    for i, h in enumerate(retrieve(query), 1):
        print(f"{i}. [{h['citation']}] (distance {h['distance']:.3f})")
        print(f"   {h['text'][:200]}...\n")


if __name__ == "__main__":
    main()