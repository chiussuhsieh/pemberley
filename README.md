# Pemberley

A voice-enabled RAG agent that lets readers converse with characters from Jane
Austen's *Pride and Prejudice*. Every response is grounded in the source text
and carries a citation to a specific volume and chapter, so the character can
never invent events that aren't in the novel.

The project is a working demonstration of problems that voice-agent systems face
in production: **hallucination control**, **persona consistency**, **grounded
generation with verifiable citations**, and **multi-agent orchestration**.

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
- [x] **Persona layer** — LangGraph state machine with citation enforcement and retry
- [x] **Multi-turn conversation** — in-session memory with query rewriting for retrieval
- [x] **Multi-agent** — multiple characters (Elizabeth, Darcy) with a router
- [ ] **Voice loop** — Deepgram STT + TTS with latency profiling
- [ ] **Deployment** — public demo

## Architecture

```
pemberley/
├── src/
│   ├── clean.py         # Strip Gutenberg boilerplate, illustrations, front matter
│   ├── chunk.py         # Chapter-aware chunking with continuity validation
│   ├── config.py        # Single source of truth for embedding function + paths
│   ├── index.py         # Embed chunks and write to Chroma
│   ├── retrieve.py      # Query the vector store, return passages with citations
│   ├── agent.py         # LangGraph conversation agent (rewrite → retrieve →
│   │                    #   generate → validate → retry/revise) + character router
│   ├── validate.py      # Citation validation (L1 format, L2 provenance)
│   ├── test_validate.py # Tests for citation validation
│   └── test_routing.py  # Tests for retry routing
├── data/                # Raw + cleaned text, chunks (git-ignored, rebuildable)
└── requirements.txt
```

Conversation flow (per turn):

```
user question
  → router      (pick the character who answers; explicit address wins)
  → rewrite     (resolve pronouns into a standalone query, for retrieval only)
  → retrieve    (semantic search, return top-k passages + Vol./Ch. citations)
  → generate    (in-character reply grounded in passages, original query kept)
  → validate    (L1: has a citation? L2: are cited chapters actually retrieved?)
       ├─ pass    → reply
       ├─ retry   → back to generate with feedback (max 2)
       └─ give up → honest in-character refusal
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

**Citation enforcement is validated, not just requested.** The system prompt asks
the model to cite, but a prompt is only a request. A validation node checks every
answer: it must contain a citation (L1), and every cited chapter must be one that
retrieval actually returned (L2). A hallucinated citation sends the turn back to
generation with feedback, up to two retries, then falls back to an honest refusal.
This turns "cited sources" into "verified sources".

**Query rewriting serves retrieval, not generation.** In multi-turn conversation
the model remembers context, but retrieval sees only the raw query, so a
follow-up like "and what about her manners?" retrieves poorly. A rewrite step
resolves references into a standalone query — but that rewritten query is used
only for retrieval. Generation still uses the original phrasing, so the
character's reply stays natural instead of sounding like a restated search query.

**The character router is conversation-level state, not graph state.** Which
character answers is decided in the main loop and persists across turns (explicit
address switches character; otherwise it stays), rather than inside the LangGraph
state, because it is a property of the conversation rather than of a single turn.

**ONNX embeddings, no PyTorch.** Uses Chroma's built-in ONNX all-MiniLM-L6-v2.
Same model as the sentence-transformers version, but with no PyTorch dependency,
which keeps the deployment image small and cold starts fast. Since the project
only needs inference, not training, the full framework buys nothing.

## Setup

Requires Python 3.11+ (developed on 3.12, Apple Silicon). Needs an Anthropic API
key in a `.env` file (`ANTHROPIC_API_KEY=...`).

```bash
git clone https://github.com/chiussuhsieh/pemberley.git
cd pemberley
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Build the index once:

```bash
python src/clean.py      # produces data/clean/pnp.txt
python src/chunk.py      # produces data/clean/chunks.jsonl (validates 61 chapters)
python src/index.py      # builds the Chroma index
```

Then talk to the characters:

```bash
python src/agent.py
```

Address a character by name to switch who answers (e.g. `Darcy, what do you
think of Wickham?`); otherwise the current character continues.

Run the tests:

```bash
python src/test_validate.py
python src/test_routing.py
```

## Tech stack

- **Language**: Python 3.12
- **Orchestration**: LangGraph (state machine with conditional routing)
- **LLM**: Anthropic API (Claude)
- **Vector store**: Chroma (PersistentClient)
- **Embeddings**: ONNX all-MiniLM-L6-v2
- **Source text**: Project Gutenberg (*Pride and Prejudice*, public domain)

Planned: Deepgram (STT), ElevenLabs/Cartesia (TTS), FastAPI (API layer),
Redis (cross-session persistence).

## License

MIT
