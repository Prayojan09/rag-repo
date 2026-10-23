"""Sentence Transformers cross-encoder passage reranking."""

from collections.abc import Sequence
from functools import lru_cache
from threading import RLock
from typing import Protocol

from app.core.config import get_settings
from app.schemas.documents import DenseSearchResult


class CrossEncoderModel(Protocol):
    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Sequence[float]: ...


class CrossEncoderReranker:
    """Score query-passage pairs with a lazily loaded cross-encoder model."""

    def __init__(self, model: CrossEncoderModel | None = None) -> None:
        settings = get_settings()
        self.model_name = settings.reranker_model
        self.model_revision = settings.reranker_model_revision
        self.device = settings.reranker_device
        self.batch_size = settings.reranker_batch_size
        self._model = model
        self._model_lock = RLock()

    def _get_model(self) -> CrossEncoderModel:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            with self._model_lock:
                if self._model is None:
                    self._model = CrossEncoder(
                        self.model_name,
                        revision=self.model_revision,
                        device=self.device,
                    )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[DenseSearchResult],
        top_k: int,
    ) -> list[DenseSearchResult]:
        if not candidates:
            return []
        pairs = [
            (query, f"{candidate.heading or ''}\n{candidate.text}") for candidate in candidates
        ]
        with self._model_lock:
            scores = self._get_model().predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        if len(scores) != len(candidates):
            raise RuntimeError("Cross-encoder returned an unexpected number of scores")
        ranked = [
            (
                float(score),
                position,
                candidate.model_copy(
                    update={
                        "retrieval_score": candidate.score,
                        "rerank_score": float(score),
                        "score": float(score),
                    }
                ),
            )
            for position, (candidate, score) in enumerate(zip(candidates, scores, strict=True))
        ]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [result for _score, _position, result in ranked[:top_k]]


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
