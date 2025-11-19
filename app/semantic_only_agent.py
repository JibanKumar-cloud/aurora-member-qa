from typing import Tuple, Dict, Any, List
from .retriever import MessageStore
from .schemas import Message
from .utility import simple_reasoner   
from .agent import BaseQAAgent


class SemanticOnlyAgent(BaseQAAgent):
    def __init__(self):
        pass

    async def answer(
        self,
        results: List[Tuple[Message, str, float]]
    ) -> Tuple[str, Dict[str, Any]]:

        if not results:
            return "I don't know.", {
                "backend": "semantic",
                "reason": "No semantic matches found",
            }

        # Use your simple_reasoner
        simple_answer, simple_reason = simple_reasoner(results)

        return simple_answer, {
            "backend": "semantic",
            "reasoning": simple_reason,
        }

