import json
import re
from pathlib import Path

CLEAN = Path("data/clean/pnp.txt")
OUT = Path("data/clean/chunks.jsonl")

# ---- Chapter segmentation ----

def roman_to_int(s):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for i, c in enumerate(s.upper()):
        v = vals[c]
        nxt = vals[s[i + 1].upper()] if i + 1 < len(s) else 0
        total += -v if v < nxt else v
    return total

def volume_of(ch):
    # Original three-volume layout: Vol I = ch 1-23, Vol II = ch 24-42, Vol III = ch 43-61
    return 1 if ch <= 23 else (2 if ch <= 42 else 3)

# ---- Sentence-boundary split with fallback ----
# Adapted from a reference implementation. The original advanced the window with
# start = break_point - overlap, which can move backwards when punctuation is
# sparse and cause a non-terminating loop. This version guarantees forward
# progress and rejects break points that would produce a tiny chunk.

def split_text(text, chunk_size=800, overlap=100):
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    floor = chunk_size // 2          # break point must sit past the window midpoint
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        bp = text.rfind(". ", start, end)
        bp = bp + 1 if bp != -1 else -1
        if bp <= start + floor:
            sp = text.rfind(" ", start, end)
            bp = sp if sp > start + floor else end
        chunks.append(text[start:bp].strip())
        nxt = bp - overlap
        start = nxt if nxt > start else bp   # enforce strictly forward progress
    return chunks

# ---- Main ----

def main():
    text = CLEAN.read_text(encoding="utf-8")

    # Case-insensitive: this edition is inconsistent, e.g. "Chapter XLVI." vs "CHAPTER XLV."
    pattern = re.compile(r"^\s*CHAPTER\s+([IVXLC]+)\.?\s*$",
                         re.MULTILINE | re.IGNORECASE)
    matches = list(pattern.finditer(text))
    print(f"chapter headings found: {len(matches)}")

    chunks = []

    # Chapter 1 has no heading in this edition: the novel opens directly after the
    # front matter. Treat everything before the first heading (from the opening
    # sentence onward) as chapter 1.
    opening = "It is a truth universally acknowledged"
    op_idx = text.find(opening)
    first_heading = matches[0].start() if matches else len(text)
    if op_idx != -1 and op_idx < first_heading:
        body = " ".join(text[op_idx:first_heading].split())
        for j, piece in enumerate(split_text(body)):
            if piece:
                chunks.append({
                    "id": f"ch01-{j:03d}", "text": piece,
                    "volume": 1, "chapter": 1, "citation": "Vol. 1, Ch. 1",
                })

    # Remaining chapters, each running from its heading to the next.
    for idx, m in enumerate(matches):
        ch = roman_to_int(m.group(1))
        vol = volume_of(ch)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = " ".join(text[start:end].split())
        for j, piece in enumerate(split_text(body)):
            if piece:
                chunks.append({
                    "id": f"ch{ch:02d}-{j:03d}", "text": piece,
                    "volume": vol, "chapter": ch,
                    "citation": f"Vol. {vol}, Ch. {ch}",
                })

    # Continuity check: chapters must be the complete run 1..61 with no gaps.
    seen = sorted(set(c["chapter"] for c in chunks))
    missing = [i for i in range(1, 62) if i not in seen]
    print(f"distinct chapters: {len(seen)} (expect 61)")
    print(f"missing chapters: {missing}")

    OUT.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    print(f"total chunks: {len(chunks)}")

if __name__ == "__main__":
    main()