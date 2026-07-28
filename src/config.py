from chromadb.utils import embedding_functions

COLLECTION = "pnp"
CHROMA_PATH = "chroma_db"
CHUNKS_PATH = "data/clean/chunks.jsonl"

def get_embedding_function():
    # Chroma's built-in ONNX all-MiniLM-L6-v2. Same model as sentence-transformers
    # but no PyTorch dependency. See decision log 2026-07-23.
    return embedding_functions.ONNXMiniLM_L6_V2()