from .retriever import MessageStore
from .config import Settings
from openai import AsyncOpenAI

class BaseQAAgent:
    async def answer(self, q: str):
        raise NotImplementedError
