import re
from typing import List, Tuple

# Matches citations like [Vol. 2, Ch. 34] and captures the inside text.
CITATION_RE = re.compile(r"\[(Vol\.\s*\d+,\s*Ch\.\s*\d+)\]")


def extract_citations(answer: str) -> List[str]:
    """Pull all [Vol. X, Ch. Y] citations out of the answer text."""
    return CITATION_RE.findall(answer)


def validate_answer(answer: str, allowed: List[str]) -> Tuple[bool, str]:
    """Check an answer's citations against the passages that were retrieved.

    L1 (format): the answer must contain at least one citation.
    L2 (provenance): every cited Vol./Ch. must be one that retrieval returned.

    Returns (passed, reason). reason is empty when passed.
    """
    found = extract_citations(answer)

    # L1: must cite something
    if not found:
        return False, "no citation found in answer"

    # L2: every citation must come from the retrieved set
    allowed_set = set(allowed)
    invalid = [c for c in found if c not in allowed_set]
    if invalid:
        return False, f"hallucinated citations not in retrieved set: {invalid}"

    return True, ""