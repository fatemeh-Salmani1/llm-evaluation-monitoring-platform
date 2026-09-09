from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from src.evaluation.metrics import (
    DeterministicEvaluationResult,
    evaluate_deterministically,
)
from src.evaluation.models import (
    BenchmarkCase,
    Difficulty,
    EvaluationCategory,
)
from src.generation.answerer import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    generate_grounded_answer,
)
from src.retrieval.embedder import DEFAULT_EMBEDDING_MODEL
from src.retrieval.run_retrieval import (
    DEFAULT_TOP_K,
    retrieve_from_files,
)


class EvaluationStatus(StrEnum):
    """Outcome of an evaluation-case execution."""

    SUCCESS = "success"
    FAILED = "failed"


class EvaluationRunRecord(BaseModel):
    """Monitoring record produced for one benchmark case."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    case_id: str = Field(pattern=r"^eval-\d{4}$")
    status: EvaluationStatus
    started_at: datetime
    duration_ms: float = Field(ge=0.0)
    question: str
    category: EvaluationCategory
    difficulty: Difficulty
    embedding_model: str
    generation_model: str
    retrieved_chunk_ids: list[str]
    retrieval_scores: list[float]
    answer: str | None = None
    metrics: DeterministicEvaluationResult | None = None
    error_message: str | None = None


def run_evaluation_case(
    case: BenchmarkCase,
    chunks_path: Path,
    embeddings_path: Path,
    client: OpenAI,
    run_id: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    top_k: int = DEFAULT_TOP_K,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> EvaluationRunRecord:
    """Run retrieval, generation, and scoring for one case."""

    resolved_run_id = run_id or str(uuid4())
    started_at = datetime.now(UTC)
    start_time = perf_counter()
    retrieved_chunk_ids: list[str] = []
    retrieval_scores: list[float] = []

    try:
        retrieved_chunks = retrieve_from_files(
            question=case.question,
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            client=client,
            model=embedding_model,
            top_k=top_k,
        )

        retrieved_chunk_ids = [
            result.chunk.chunk_id
            for result in retrieved_chunks
        ]
        retrieval_scores = [
            result.similarity_score
            for result in retrieved_chunks
        ]

        grounded_answer = generate_grounded_answer(
            question=case.question,
            retrieved_chunks=retrieved_chunks,
            client=client,
            model=generation_model,
            max_output_tokens=max_output_tokens,
        )

        metrics = evaluate_deterministically(
            case=case,
            answer=grounded_answer.answer,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

        duration_ms = (
            perf_counter() - start_time
        ) * 1000

        return EvaluationRunRecord(
            run_id=resolved_run_id,
            case_id=case.case_id,
            status=EvaluationStatus.SUCCESS,
            started_at=started_at,
            duration_ms=duration_ms,
            question=case.question,
            category=case.category,
            difficulty=case.difficulty,
            embedding_model=embedding_model,
            generation_model=generation_model,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieval_scores=retrieval_scores,
            answer=grounded_answer.answer,
            metrics=metrics,
        )

    except (
        OpenAIError,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        duration_ms = (
            perf_counter() - start_time
        ) * 1000

        return EvaluationRunRecord(
            run_id=resolved_run_id,
            case_id=case.case_id,
            status=EvaluationStatus.FAILED,
            started_at=started_at,
            duration_ms=duration_ms,
            question=case.question,
            category=case.category,
            difficulty=case.difficulty,
            embedding_model=embedding_model,
            generation_model=generation_model,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieval_scores=retrieval_scores,
            error_message=str(error),
        )