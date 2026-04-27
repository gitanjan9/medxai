"""Chat router — POST /v1/chat

Accepts a conversation history and an optional analysis context (PrimaryPrediction
dict) and returns a medically-grounded reply.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.common.logging import get_logger
from src.serve.services.chat_service import generate_reply

router = APIRouter(prefix="/v1", tags=["chat"])
logger = get_logger("serve.chat")


class ChatMessage(BaseModel):
    role: str      # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    context: Optional[dict] = None   # serialised PrimaryPrediction


class ChatResponse(BaseModel):
    reply: str
    model_used: str


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    ctx = req.context or {}
    logger.info("chat turns=%d  label=%s", len(messages), ctx.get("label", "n/a"))
    reply, model_used = generate_reply(messages, ctx)
    return ChatResponse(reply=reply, model_used=model_used)
