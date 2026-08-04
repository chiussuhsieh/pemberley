from agent import route_after_validate, MAX_RETRIES

cases = [
    # (description, state, expected_route)
    ("pass: no failure reason",
     {"reason": "", "retry_count": 0}, "pass"),

    ("retry: failure with retries left",
     {"reason": "no citation found", "retry_count": 0}, "retry"),

    ("retry: failure on second attempt",
     {"reason": "hallucinated citation", "retry_count": 1}, "retry"),

    ("give_up: failure after max retries",
     {"reason": "still failing", "retry_count": MAX_RETRIES}, "give_up"),
]

for desc, state, expected in cases:
    result = route_after_validate(state)
    mark = "OK" if result == expected else "WRONG"
    print(f"[{mark}] {desc} -> {result}")