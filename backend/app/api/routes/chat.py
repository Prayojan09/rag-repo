"""Evidence-grounded question answering endpoint."""

import logging
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import APIPrincipal, require_api_token
from app.db.models import Conversation, ConversationMessage
from app.db.session import get_db
from app.generation.providers import LLMProviderError
from app.retrieval.dense import get_dense_vector_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import answer_question

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"],
    dependencies=[Depends(require_api_token)],
)
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse)
def chat(
    http_request: Request,
    request: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> ChatResponse:
    try:
        conversation = None
        if request.conversation_id:
            conversation = db.scalar(
                select(Conversation).where(
                    Conversation.id == request.conversation_id,
                    Conversation.tenant_id == principal.tenant_id,
                )
            )
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            conversation = Conversation(
                id=str(uuid4()),
                tenant_id=principal.tenant_id,
                title=request.question[:200],
            )
            db.add(conversation)
            db.flush()
        history_rows = db.scalars(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation.id)
            .order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc())
            .limit(8)
        ).all()
        conversation_history = [
            {"role": message.role, "content": message.content[:2000]}
            for message in reversed(history_rows)
            if message.role in {"user", "assistant"}
        ]
        db.add(
            ConversationMessage(
                id=str(uuid4()),
                conversation_id=conversation.id,
                role="user",
                content=request.question,
            )
        )
        db.commit()
        vector_service = get_dense_vector_service() if request.mode != "keyword" else None
        answer = answer_question(
            db,
            request,
            vector_service=vector_service,
            tenant_id=principal.tenant_id,
            conversation_history=conversation_history,
        )
        answer.conversation_id = conversation.id
        db.add(
            ConversationMessage(
                id=str(uuid4()),
                conversation_id=conversation.id,
                role="assistant",
                content=answer.answer,
                response_data=answer.model_dump(mode="json"),
            )
        )
        conversation.updated_at = datetime.now(UTC)
        db.commit()
        return answer
    except HTTPException:
        raise
    except Exception as exc:
        # Do not log upstream prompts or model outputs: they can include document text.
        request_id = getattr(http_request.state, "request_id", "unknown")
        cause = exc.__cause__
        provider_error = str(cause) if isinstance(cause, LLMProviderError) else "n/a"
        validation_fields = "n/a"
        if isinstance(cause, ValidationError):
            validation_fields = ",".join(
                ".".join(str(part) for part in error["loc"])
                + ":"
                + str(error["type"])
                for error in cause.errors(
                    include_input=False, include_context=False, include_url=False
                )
            )[:500]
        logger.error(
            "Chat generation failed (request_id=%s error_type=%s cause_type=%s "
            "provider_error=%s validation_fields=%s)",
            request_id,
            type(exc).__name__,
            type(cause).__name__ if cause else "none",
            provider_error,
            validation_fields,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Question answering is temporarily unavailable. Check the backend "
                f"terminal for request ID {request_id}."
            ),
        ) from exc
