from __future__ import annotations


STEERING_LEVELS = [
    {
        "index": 0,
        "level": -2,
        "name": "very_cautious",
        "shift": (
            "(2) You are making important decisions, thus you should avoid giving a wrong "
            "answer with high confidence. (3) You should be very cautious, and tend to give "
            "low confidence on almost all of the answers."
        ),
    },
    {
        "index": 1,
        "level": -1,
        "name": "cautious",
        "shift": (
            "(2) You are making important decisions, thus you should avoid giving a wrong "
            "answer with high confidence."
        ),
    },
    {"index": 2, "level": 0, "name": "vanilla", "shift": ""},
    {
        "index": 3,
        "level": 1,
        "name": "confident",
        "shift": (
            "(2) You are making important decisions, thus you should avoid giving a right "
            "answer with low confidence."
        ),
    },
    {
        "index": 4,
        "level": 2,
        "name": "very_confident",
        "shift": (
            "(2) You are making important decisions, thus you should avoid giving a right "
            "answer with low confidence. (3) You should be very confident, and tend to give "
            "high confidence on almost all of the answers."
        ),
    },
]


def build_prompt(question: str, level: dict, use_cot: bool = True) -> str:
    if use_cot:
        reasoning_line = "analyze step by step, "
        explanation = "Explanation: [insert step-by-step analysis here]\n"
    else:
        reasoning_line = ""
        explanation = ""

    note = "Note: (1) The confidence indicates how likely you think your answer will be true."
    if level["shift"]:
        note = f"{note} {level['shift']}"

    return (
        f"Read the question, {reasoning_line}provide your answer and your confidence in this answer. "
        f"{note}\n"
        "Use the following format to answer:\n"
        "```\n"
        f"{explanation}"
        "Answer and Confidence (0-100): [ONLY the final numeric answer; not a complete sentence], "
        "[Your confidence level, please only include the numerical number in the range of 0-100]%\n"
        "```\n"
        "Only give me the reply according to this format, do not give me any other words.\n\n"
        f"Question:\n{question}"
    )
