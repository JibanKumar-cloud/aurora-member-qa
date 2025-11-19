from typing import Tuple, List
from .schemas import Message


def simple_reasoner(
    results: List[tuple]
) -> Tuple[str, str]:
    """
    Local reasoning fallback for semantic search.
    Returns (answer, explanation_string)
    """

    if not results:
        return (
            "I couldn’t find any relevant information in the member messages.",
            "No messages were retrieved as relevant for this question."
        )

    top_msg, _text, score = results[0]

    answer = (
        f"Based on the messages, the best matching information is:\n"
        f"\"{top_msg.text}\""
    )

    reasoning = (
        "I ranked all messages by semantic similarity to the question and "
        "selected the top one as the most relevant."
    )

    return answer, reasoning
