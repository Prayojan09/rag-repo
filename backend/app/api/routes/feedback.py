"""Persist lightweight user ratings for generated answers."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.security import APIPrincipal, require_api_token
from app.db.models import ChatFeedback
from app.db.session import get_db
from app.schemas.feedback import FeedbackCreate, FeedbackResponse

router = APIRouter(
    prefix="/api/v1/feedback",
    tags=["feedback"],
    dependencies=[Depends(require_api_token)],
)


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_feedback(
    request: FeedbackCreate,
    db: Annotated[Session, Depends(get_db)],
    principal: Annotated[APIPrincipal, Depends(require_api_token)],
) -> ChatFeedback:
    feedback = ChatFeedback(
        id=str(uuid4()),
        tenant_id=principal.tenant_id,
        answer_id=str(request.answer_id),
        rating=request.rating,
        comment=request.comment,
    )
    try:
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback could not be saved",
        ) from exc
    return feedback
