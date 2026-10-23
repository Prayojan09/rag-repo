"""Tenant-scoped conversation history endpoints."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.security import APIPrincipal, require_api_token
from app.db.models import Conversation
from app.db.session import get_db
from app.schemas.conversations import ConversationDetail, ConversationList, ConversationSummary

router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["conversations"],
    dependencies=[Depends(require_api_token)],
)


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
def create_conversation(
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
    title: Annotated[str, Query(min_length=1, max_length=200)] = "New conversation",
) -> Conversation:
    conversation = Conversation(
        id=str(uuid4()), tenant_id=principal.tenant_id, title=title.strip() or "New conversation"
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.get("", response_model=ConversationList)
def list_conversations(
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationList:
    criteria = Conversation.tenant_id == principal.tenant_id
    total = db.scalar(select(func.count()).select_from(Conversation).where(criteria)) or 0
    items = db.scalars(
        select(Conversation)
        .where(criteria)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return ConversationList(items=items, total=total, limit=limit, offset=offset)


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> Conversation:
    conversation = db.scalar(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == principal.tenant_id,
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> None:
    conversation = db.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == principal.tenant_id,
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conversation)
    db.commit()
