from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import json
import os
from .retriever import MessageStore
from .config import Settings
from openai import OpenAI
from .utility import simple_reasoner
from .schemas import Message
from .agent import BaseQAAgent

# If OPENAI_API_KEY is not set, we'll fallback to local reasoning.
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_client: Optional[OpenAI] = OpenAI() if _OPENAI_KEY else None


SYSTEM_PROMPT = """
You are an AI assistant for Aurora that answers questions about member messages.

You are given:
- A user question.
- A small set of messages from Aurora's member message history.

Your job:
1. Read ALL messages carefully and decide which are relevant.
2. Infer the best possible answer that is directly supported by the messages.
3. If the answer requires combining multiple messages, do so and explain your reasoning.
4. If the messages do NOT contain enough information to answer confidently, say you cannot answer based on the messages.

Important rules:
- ONLY use the information in the provided messages. Do not make up facts.
- Be concise and factual.
- If there is ambiguity, say so explicitly.
- Answer in natural language, but stay precise.
"""

USER_TEMPLATE = """
Question:
{question}

Relevant messages:
{messages_block}

Instructions:
- First, think through the messages and decide what they imply.
- Then respond in JSON only with exactly these keys:
  - "answer": a short natural-language answer to the question
  - "reasoning": 2-3 sentence explanation of how you used the messages, or
                 why you could not answer

Your entire reply MUST be a valid JSON object, nothing else.
Example:
{{
  "answer": "They have 2 cars.",
  "reasoning": "Message M1 says 'I currently have two cars in the garage', which directly answers the question."
}}
"""


def _format_messages(results: List[Tuple[Message, str, float]]) -> str:
    """
    Turn (Message, text_repr, score) into a readable context block.
    """
    lines = []
    for msg, _text, score in results:
        lines.append(
            f"- id={msg.id} | user={msg.member_name or msg.member_id} | score={score:.3f}\n"
            f"  message: {msg.text}"
        )
    return "\n".join(lines)


class OpenAIQAAgent(BaseQAAgent):
    """
    LLM-based agent that reasons over retrieved messages, with a robust
    local fallback if the LLM call fails or no API key is configured.
    """

    def __init__(self, model_name: str = "gpt-4.1-mini") -> None:
        self.model_name = model_name

    def _call_llm(
        self,
        question: str,
        results: List[Tuple[Message, str, float]],
    ) -> Dict[str, Any]:
        if _client is None:
            raise RuntimeError("OPENAI_API_KEY is not set; using local reasoning instead.")

        messages_block = _format_messages(results)
      
        user_content = USER_TEMPLATE.format(
            question=question,
            messages_block=messages_block,
        )
        print(user_content)
        resp = _client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_content.strip()},
            ],
            response_format={"type": "json_object"},
        )

        raw_content = resp.choices[0].message.content
        print(raw_content)
        parsed = json.loads(raw_content)

        return {
            "answer": parsed.get("answer", "").strip(),
            "reasoning": parsed.get("reasoning", "").strip(),
        }

    def answer(
        self,
        question: str,
        results: List[Tuple[Message, str, float]],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (answer, reasoning_dict).
        - Tries LLM if possible.
        - Falls back to local reasoning in any error case.
        """
        if not results:
            return (
                "I couldn’t find any relevant information in the member messages.",
                {
                    "backend": "openai",
                    "reason": "No messages were retrieved as relevant for this question.",
                },
            )

        try:
            parsed = self._call_llm(question, results)
            answer = parsed.get("answer") or "I couldn’t produce an answer."
            reasoning_text = parsed.get("reasoning") or ""
            reasoning: Dict[str, Any] = {
                "backend": "openai",
                "explanation": reasoning_text,
            }
            return answer, reasoning

        except Exception as e:
            try:
                print("[OpenAIQAAgent] LLM error repr:", repr(e))
            except Exception:
                pass

            # Fallback: local deterministic reasoning
            simple_answer, simple_reason = simple_reasoner(results)
            return simple_answer, {
                "backend": "openai-fallback fetching semantic search",
                "explanation": simple_reason,
            }
        