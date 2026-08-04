from validate import validate_answer

# Simulate what retrieve returned for a query.
retrieved = ["Vol. 1, Ch. 16", "Vol. 1, Ch. 5", "Vol. 2, Ch. 34", "Vol. 2, Ch. 41"]

cases = [
    # (description, answer, expected_pass)
    ("valid: cites a retrieved chapter",
     "He spoke of his scruples [Vol. 2, Ch. 34].", True),

    ("valid: multiple good citations",
     "As noted [Vol. 1, Ch. 16] and later [Vol. 2, Ch. 34].", True),

    ("fail L1: no citation at all",
     "Mr. Darcy is a proud and disagreeable man.", False),

    ("fail L2: hallucinated citation",
     "He confessed everything [Vol. 3, Ch. 50].", False),

    ("fail L2: one good one bad",
     "True [Vol. 1, Ch. 5], and also [Vol. 3, Ch. 99].", False),
]

for desc, answer, expected in cases:
    passed, reason = validate_answer(answer, retrieved)
    mark = "OK" if passed == expected else "WRONG"
    print(f"[{mark}] {desc}")
    if not passed:
        print(f"       reason: {reason}")