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

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(body, encoding="utf-8")
print(f"final: {len(body)} chars")
print("---")
print(body[:300])