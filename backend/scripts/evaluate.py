"""Run reproducible retrieval metrics against the versioned evaluation corpus."""

import argparse
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from qdrant_client import QdrantClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.models import Base, Document, DocumentChunk  # noqa: E402
from app.evaluation.metrics import mean_metric, retrieval_metrics  # noqa: E402
from app.retrieval.dense import DenseVectorService  # noqa: E402
from app.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from app.schemas.chat import ChatRequest  # noqa: E402
from app.services.chat import answer_question  # noqa: E402
from app.services.hybrid_search import search  # noqa: E402

DATASET_PATH = BACKEND_ROOT / "evaluation" / "dataset.json"
MODES = ("dense", "keyword", "hybrid", "hybrid_reranked")


def _stable_id(name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"rag-evaluation:{name}"))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _load_corpus(db: Session, dataset: dict) -> dict[str, str]:
    chunk_ids: dict[str, str] = {}
    documents_by_file: dict[str, Document] = {}
    for document_data in dataset["documents"]:
        filename = document_data["filename"]
        document = Document(
            id=_stable_id(f"doc:{filename}"),
            filename=filename,
            media_type="text/plain",
            sha256=sha256(filename.encode()).hexdigest(),
            size_bytes=1,
            version=1,
            status="ready",
            storage_key=f"eval-{Path(filename).stem}.txt",
        )
        db.add(document)
        documents_by_file[filename] = document
        for chunk_data in document_data["chunks"]:
            chunk_id = _stable_id(f"chunk:{chunk_data['id']}")
            chunk_ids[chunk_data["id"]] = chunk_id
            db.add(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document.id,
                    ordinal=len(chunk_ids),
                    text=chunk_data["text"],
                    page_number=chunk_data["page_number"],
                    heading=chunk_data["heading"],
                )
            )
    db.commit()
    return chunk_ids


def run_retrieval_benchmark(args: argparse.Namespace) -> dict:
    dataset_bytes = DATASET_PATH.read_bytes()
    dataset = json.loads(dataset_bytes)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    settings = get_settings()
    client = QdrantClient(":memory:")
    dense_service = None
    reranker = None
    started_setup = time.perf_counter()

    try:
        with Session(engine, autoflush=False, expire_on_commit=False) as db:
            chunk_ids = _load_corpus(db, dataset)
            if any(mode in args.modes for mode in ("dense", "hybrid", "hybrid_reranked")):
                dense_service = DenseVectorService(client=client)
                # Trigger model loading and build the real in-memory vector index
                # before timing individual warm retrieval queries.
                dense_service.embedder.dimension()
                for document in db.query(Document).order_by(Document.filename).all():
                    dense_service.index_document(
                        document.id,
                        document.filename,
                        document.version,
                        document.chunks,
                    )
            if "hybrid_reranked" in args.modes:
                reranker = CrossEncoderReranker()
                reranker._get_model()
            setup_seconds = time.perf_counter() - started_setup

            per_mode: dict[str, list[dict]] = {mode: [] for mode in args.modes}
            for query in dataset["queries"]:
                relevant = {chunk_ids[item] for item in query["relevant_chunk_ids"]}
                for mode in args.modes:
                    candidate_depth = args.k
                    search_mode = mode
                    if mode == "hybrid_reranked":
                        search_mode = "hybrid"
                        candidate_depth = min(
                            max(args.k * settings.search_candidate_multiplier,
                                settings.reranker_candidate_k),
                            settings.search_candidate_max,
                        )
                    start = time.perf_counter()
                    results = search(
                        db,
                        query["question"],
                        candidate_depth,
                        mode=search_mode,
                        candidate_multiplier=settings.search_candidate_multiplier,
                        candidate_max=settings.search_candidate_max,
                        rrf_constant=settings.rrf_constant,
                        vector_service=dense_service,
                    )
                    if mode == "hybrid_reranked":
                        results = reranker.rerank(query["question"], results, args.k)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    ranked_ids = [result.chunk_id for result in results[: args.k]]
                    metrics = retrieval_metrics(ranked_ids, relevant, args.k)
                    per_mode[mode].append(
                        {
                            "query_id": query["id"],
                            **metrics,
                            "latency_ms": round(elapsed_ms, 3),
                            "result_count": len(ranked_ids),
                            "abstention_case": query["should_abstain"],
                        }
                    )

            aggregate = {}
            for mode, cases in per_mode.items():
                latencies = [case["latency_ms"] for case in cases]
                answerable = [case for case in cases if case["recall_at_k"] is not None]
                aggregate[mode] = {
                    "answerable_query_count": len(answerable),
                    "recall_at_k": mean_metric(cases, "recall_at_k"),
                    "mrr_at_k": mean_metric(cases, "mrr_at_k"),
                    "ndcg_at_k": mean_metric(cases, "ndcg_at_k"),
                    "mean_latency_ms": round(statistics.mean(latencies), 3)
                    if latencies else None,
                    "p95_latency_ms": round(_percentile(latencies, 0.95), 3)
                    if latencies else None,
                    "queries_with_results": sum(case["result_count"] > 0 for case in cases),
                }

            return {
                "dataset_id": dataset["dataset_id"],
                "dataset_sha256": sha256(dataset_bytes).hexdigest(),
                "dataset_note": dataset["description"],
                "created_at_utc": datetime.now(UTC).isoformat(),
                "k": args.k,
                "embedding_model": dense_service.model_name if dense_service else None,
                "reranker_model": reranker.model_name if reranker else None,
                "setup_seconds_including_model_load_and_index": round(setup_seconds, 3),
                "per_mode": aggregate,
                "per_query": per_mode,
                "answer_evaluation": None,
                "cost_estimate": None,
                "notes": [
                    "Synthetic smoke dataset; results are not evidence of production quality.",
                    "Retrieval latency excludes model load/index time, reported separately.",
                    "Dense uses configured Sentence Transformers and in-memory Qdrant.",
                    "Answer evaluation and provider costs require explicit --include-chat opt-in.",
                ],
            }
    finally:
        client.close()
        engine.dispose()


def run_chat_benchmark(args: argparse.Namespace, result: dict) -> None:
    """Run cost-bearing chat calls only when the caller opts in on the command line."""
    from app.generation.providers import get_llm_provider

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    client = QdrantClient(":memory:")
    try:
        with Session(engine, autoflush=False, expire_on_commit=False) as db:
            chunk_ids = _load_corpus(db, dataset)
            dense_service = DenseVectorService(client=client)
            dense_service.embedder.dimension()
            for document in db.query(Document).all():
                dense_service.index_document(
                    document.id, document.filename, document.version, document.chunks
                )
            reranker = CrossEncoderReranker()
            reranker._get_model()
            provider = get_llm_provider()
            rows = []
            for query in dataset["queries"]:
                started = time.perf_counter()
                response = answer_question(
                    db,
                    ChatRequest(
                        question=query["question"],
                        mode="hybrid",
                        top_k=args.k,
                        rerank=True,
                    ),
                    vector_service=dense_service,
                    provider=provider,
                    reranker=reranker,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                answer_text = response.answer.casefold()
                facts = query["answer_key_facts"]
                fact_recall = (
                    sum(fact.casefold() in answer_text for fact in facts) / len(facts)
                    if facts else None
                )
                relevant = {chunk_ids[item] for item in query["relevant_chunk_ids"]}
                citations = {citation.chunk_id for citation in response.citations}
                citation_precision = (
                    len(citations & relevant) / len(citations) if citations else None
                )
                rows.append(
                    {
                        "query_id": query["id"],
                        "should_abstain": query["should_abstain"],
                        "did_abstain": response.abstained,
                        "abstention_correct": response.abstained == query["should_abstain"],
                        "key_fact_recall": fact_recall,
                        "citation_precision_against_gold": citation_precision,
                        "evidence_verified": response.evidence_verified,
                        "unverified_claim_count": response.unverified_claim_count,
                        "verified_claim_count": len(response.claims),
                        "prompt_tokens": response.llm_prompt_tokens,
                        "completion_tokens": response.llm_completion_tokens,
                        "latency_ms": round(elapsed_ms, 3),
                        "answer": response.answer,
                        "citation_chunk_ids": sorted(citations),
                    }
                )

            has_rates = (
                args.input_cost_per_1m is not None
                and args.output_cost_per_1m is not None
            )
            for row in rows:
                if (
                    has_rates
                    and row["prompt_tokens"] is not None
                    and row["completion_tokens"] is not None
                ):
                    row["estimated_cost"] = (
                        row["prompt_tokens"] * args.input_cost_per_1m
                        + row["completion_tokens"] * args.output_cost_per_1m
                    ) / 1_000_000
                else:
                    row["estimated_cost"] = None

            relevant_cases = [row for row in rows if row["key_fact_recall"] is not None]
            answerable_citation_cases = [
                row for row in rows
                if row["citation_precision_against_gold"] is not None
            ]
            claims_total = sum(
                row["verified_claim_count"] + row["unverified_claim_count"] for row in rows
            )
            unverified_total = sum(row["unverified_claim_count"] for row in rows)
            result["answer_evaluation"] = {
                "provider": provider.provider_name,
                "model": provider.model_name,
                "query_count": len(rows),
                "key_fact_recall": mean_metric(relevant_cases, "key_fact_recall"),
                "citation_precision_against_gold": mean_metric(
                    answerable_citation_cases, "citation_precision_against_gold"
                ),
                "abstention_accuracy": mean_metric(
                    [{"abstention_accuracy": float(row["abstention_correct"])} for row in rows],
                    "abstention_accuracy",
                ),
                "unverified_claim_rate": unverified_total / claims_total if claims_total else None,
                "mean_latency_ms": round(statistics.mean(row["latency_ms"] for row in rows), 3),
                "p95_latency_ms": round(
                    _percentile([row["latency_ms"] for row in rows], 0.95), 3
                ),
                "mean_prompt_tokens": mean_metric(rows, "prompt_tokens"),
                "mean_completion_tokens": mean_metric(rows, "completion_tokens"),
                "mean_estimated_cost": mean_metric(rows, "estimated_cost"),
                "rates_per_million_tokens": {
                    "input": args.input_cost_per_1m,
                    "output": args.output_cost_per_1m,
                },
                "per_query": rows,
                "metric_note": (
                    "Key fact recall uses exact substring matching to dataset facts; "
                    "it is not a complete semantic correctness judge."
                ),
            }
            result["cost_estimate"] = result["answer_evaluation"]["mean_estimated_cost"]
    finally:
        client.close()
        engine.dispose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
        help="Retrieval paths to compare (dense, keyword, hybrid, hybrid_reranked).",
    )
    parser.add_argument("--k", type=int, default=5, help="Evaluation cutoff; default: 5.")
    parser.add_argument(
        "--include-chat",
        action="store_true",
        help="Make real LLM calls; this may incur provider charges.",
    )
    parser.add_argument(
        "--input-cost-per-1m",
        type=float,
        help="Configured provider input-token rate.",
    )
    parser.add_argument(
        "--output-cost-per-1m",
        type=float,
        help="Configured provider output-token rate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/latest.json"),
        help="Write the full report here (relative to backend/ unless absolute).",
    )
    args = parser.parse_args()
    if args.k < 1 or args.k > 100:
        parser.error("--k must be between 1 and 100")
    if (args.input_cost_per_1m is None) != (args.output_cost_per_1m is None):
        parser.error("Set both token rates or neither")
    if args.input_cost_per_1m is not None and (
        args.input_cost_per_1m < 0 or args.output_cost_per_1m < 0
    ):
        parser.error("Token rates cannot be negative")
    if not args.include_chat and (args.input_cost_per_1m is not None):
        parser.error("Token rates are only used with --include-chat")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run_retrieval_benchmark(args)
        if args.include_chat:
            run_chat_benchmark(args, report)
    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    output_path = args.output
    if not output_path.is_absolute():
        output_path = BACKEND_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "dataset_id", "dataset_sha256", "k", "per_mode", "answer_evaluation", "cost_estimate"
    )}, indent=2))
    print(f"Full report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
