# Pemberley

A voice-enabled RAG agent that lets readers converse with characters from Jane
Austen's *Pride and Prejudice*. Every response is grounded in the source text
and carries a citation to a specific volume and chapter, so the character can
never invent events that aren't in the novel.

The project is a working demonstration of three problems that voice-agent
systems face in production: **hallucination control**, **persona consistency**,
and **grounded generation with verifiable citations**.

## Why this project

Character role-play with LLMs has two failure modes. The model **hallucinates**,
inventing plot points the source never contains, and it suffers **persona
drift**, saying things out of character or out of period. Pemberley addresses
both by forcing every answer to be traceable to a specific passage: if the model
can't ground a claim in retrieved text, it declines or rephrases rather than
inventing.

This is the same class of problem that voice-agent companies solve every day,
which is why the stack deliberately builds on their tooling (Deepgram for STT,
ElevenLabs/Cartesia for TTS in later phases).

## Current status

This is an in-progress project built in weekly milestones.

- [x] **Ingestion pipeline** — clean, chapter-aware chunking, embedding, retrieval
- [x] **Citation grounding** — every retrieved chunk carries a Vol./Ch. citation
- [ ] **Persona layer** — LangGraph state machine enforcing character + citations
- [ ] **Voice loop** — Deepgram STT + TTS with latency profiling
- [ ] **Deployment** — public demo

## Architecture

```
pemberley/
├── src/
│   ├── clean.py      # Strip Gutenberg boilerplate, illustrations, front matter
│   ├── chunk.py      # Chapter-aware chunking with continuity validation
│   ├── config.py     # Single source of truth for embedding function + paths
│   ├── index.py      # Embed chunks and write to Chroma
│   └── retrieve.py   # Query the vector store, return passages with citations
├── data/             # Raw + cleaned text, chunks (git-ignored, rebuildable)
└── requirements.txt
```

Data flow:

```
Gutenberg raw text
  → clean.py     (remove non-body content)
  → chunk.py     (split by chapter, then sentence-aligned within chapter)
  → index.py     (embed with ONNX MiniLM, store in Chroma)
  → retrieve.py  (semantic search, return top-k with Vol./Ch. citations)
```

## Design decisions

The citation guarantee constrains every upstream choice. A few that came out of
building it:

**Chunks never cross chapter boundaries.** Because every answer must cite a
chapter, a chunk that spans two chapters would produce a wrong citation. So
chunking happens in two levels: split the book into chapters first, then split
within each chapter. This also forced the cleaning step to remove all non-body
content (license boilerplate, illustration blocks, the table of contents),
since none of it belongs to a chapter and can't be assigned a citation.

**Metadata is semantic, not positional.** Chunks store volume and chapter, not a
chunk index. A chunk index only means something inside this particular file; a
reader can't use it to verify anything. "Vol. II, Ch. 11" points to the same
passage in any edition, which is what makes the citation actually verifiable.

**Chapter segmentation is validated, not hard-coded.** This edition has three
irregularities: a leftover illustrations list, a first chapter with no heading,
and one chapter whose heading uses different capitalization. Rather than
hard-coding fixes for specific chapters, the pipeline validates that the output
is the complete run of chapters 1–61 and reports any gap, so future texts will
surface their own irregularities instead of failing silently.

**ONNX embeddings, no PyTorch.** Uses Chroma's built-in ONNX all-MiniLM-L6-v2.
Same model as the sentence-transformers version, but with no PyTorch dependency,
which keeps the deployment image small and cold starts fast. Since the project
only needs inference, not training, the full framework buys nothing.

## Setup

Requires Python 3.11+ (developed on 3.12, Apple Silicon).

```bash
git clone https://github.com/chiussuhsieh/pemberley.git
cd pemberley
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the ingestion pipeline in order:

```bash
python src/clean.py      # produces data/clean/pnp.txt
python src/chunk.py      # produces data/clean/chunks.jsonl (validates 61 chapters)
python src/index.py      # builds the Chroma index
```

Then query:

```bash
python src/retrieve.py "What does Darcy say about his own pride?"
```

Returns the top passages, each with its volume/chapter citation and semantic
distance.

## Tech stack

- **Language**: Python 3.12
- **Vector store**: Chroma (PersistentClient)
- **Embeddings**: ONNX all-MiniLM-L6-v2
- **Source text**: Project Gutenberg (*Pride and Prejudice*, public domain)

Planned: LangGraph (persona + citation enforcement), Deepgram (STT),
ElevenLabs/Cartesia (TTS), FastAPI (API layer).

## License

MIT
