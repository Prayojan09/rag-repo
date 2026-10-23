"""Ranking metrics used by the retrieval evaluation runner."""

import math
from collections.abc import Sequence


def retrieval_metrics(
    ranked_chunk_ids: Sequence[str], relevant_chunk_ids: set[str], k: int
) -> dict[str, float | None]:
    """Compute Recall@K, reciprocal rank, and binary nDCG@K for one query."""
    if not relevant_chunk_ids:
        return {"recall_at_k": None, "mrr_at_k": None, "ndcg_at_k": None}
    ranked = list(ranked_chunk_ids[:k])
    hits = [1 if chunk_id in relevant_chunk_ids else 0 for chunk_id in ranked]
    recall = sum(hits) / len(relevant_chunk_ids)
    reciprocal_rank = next((1 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant_chunk_ids), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "recall_at_k": recall,
        "mrr_at_k": reciprocal_rank,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def mean_metric(cases: Sequence[dict[str, float | None]], key: str) -> float | None:
    values = [case[key] for case in cases if case.get(key) is not None]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)
