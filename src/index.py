import json
from pathlib import Path

import chromadb

from config import COLLECTION, CHROMA_PATH, CHUNKS_PATH, get_embedding_function


def main():
    rows = [json.loads(l) for l in Path(CHUNKS_PATH).read_text().splitlines() if l.strip()]
    print(f"loaded {len(rows)} chunks")

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Rebuild from scratch so re-running is idempotent (deterministic IDs also
    # allow upsert, but a clean rebuild avoids stale chunks after re-chunking).
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = client.create_collection(
        name=COLLECTION,
        embedding_function=get_embedding_function(),
    )

    # Chroma embeds in batches internally; add() takes the whole list at once.
    col.add(
        ids=[r["id"] for r in rows],
        documents=[r["text"] for r in rows],
        metadatas=[
            {"volume": r["volume"], "chapter": r["chapter"], "citation": r["citation"]}
            for r in rows
        ],
    )
    print(f"indexed {col.count()} chunks into '{COLLECTION}'")


if __name__ == "__main__":
    main()