"""Conversation history API schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    response_data: dict | None
    created_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[ConversationMessageResponse]


class ConversationList(BaseModel):
    items: list[ConversationSummary]
    total: int
    limit: int
    offset: int
