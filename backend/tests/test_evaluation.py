import math

import pytest

from app.evaluation.metrics import mean_metric, retrieval_metrics


def test_retrieval_metrics_compute_recall_mrr_and_ndcg():
    metrics = retrieval_metrics(
        ["irrelevant", "relevant-b", "relevant-a"],
        {"relevant-a", "relevant-b"},
        3,
    )

    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr_at_k"] == 0.5
    assert metrics["ndcg_at_k"] == pytest.approx(
        (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )


def test_retrieval_metrics_exclude_unanswerable_queries():
    assert retrieval_metrics(["a"], set(), 5) == {
        "recall_at_k": None,
        "mrr_at_k": None,
        "ndcg_at_k": None,
    }
    assert mean_metric([{"recall_at_k": None}], "recall_at_k") is None
