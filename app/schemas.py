from __future__ import annotations

from typing import Optional, Any, Dict
from pydantic import BaseModel


class Message(BaseModel):
    id: Any
    member_id: Optional[Any] = None
    member_name: Optional[str] = None
    text: str


class AnswerResponse(BaseModel):
    answer: str
    # Optional reasoning/debug info (intent, member, top hits)
    reasoning: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
