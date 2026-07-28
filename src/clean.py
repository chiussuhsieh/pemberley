import re
from pathlib import Path

RAW = Path("data/raw/pnp.txt")
OUT = Path("data/clean/pnp.txt")

text = RAW.read_text(encoding="utf-8-sig")

start = re.search(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*$",
                  text, re.MULTILINE)
end = re.search(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*$",
                text, re.MULTILINE)

if not start or not end:
    raise SystemExit("Gutenberg markers not found, inspect the raw file")

body = text[start.end():end.start()].strip()
print(f"after header/footer strip: {len(body)} chars")

ILLUS = re.compile(r"\[Illustration[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", re.DOTALL)
body, n_illus = ILLUS.subn("", body)
print(f"removed {n_illus} illustration blocks")
print(f"remaining markers: {body.count('[Illustration')}")

# Remove the "List of Illustrations" front-matter block.
# This edition places it between the title page and Chapter 1, using lines like
# "Heading to Chapter I. ... <page>". The block ends right before the novel's
# opening sentence.
opening = "It is a truth universally acknowledged"
op_idx = body.find(opening)
il_idx = body.lower().find("list of illustrations")
if il_idx != -1 and op_idx != -1 and il_idx < op_idx:
    removed = op_idx - il_idx
    body = body[:il_idx] + body[op_idx:]
    print(f"removed illustrations list: {removed} chars")
else:
    print("WARNING: illustrations list markers not found as expected")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(body, encoding="utf-8")
print(f"final: {len(body)} chars")
print("---")
print(body[:300])