from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.judge import JudgeScores
from src.evaluation.models import (
    BenchmarkCase,
    Difficulty,
    EvaluationCategory,
)
from src.evaluation.runner import (
    EvaluationStatus,
    run_evaluation_case,
)
from src.retrieval.embedder import ChunkEmbedding
from src.retrieval.embedding_storage import write_embeddings
from src.retrieval.models import DocumentChunk
from src.retrieval.storage import write_chunks_jsonl


class FakeEmbeddings:
    """Fake embeddings endpoint for evaluation-runner tests."""

    def create(
        self,
        *,
        model: str,
        input: list[str],
    ) -> SimpleNamespace:
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[1.0, 0.0],
                )
            ]
        )


class FakeResponses:
    """Fake Responses endpoint for evaluation-runner tests."""

    def __init__(
        self,
        error_message: str | None = None,
    ) -> None:
        self.error_message = error_message
        self.create_calls = 0
        self.parse_calls = 0

    def create(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        max_output_tokens: int,
    ) -> SimpleNamespace:
        self.create_calls += 1

        if self.error_message is not None:
            raise RuntimeError(self.error_message)

        return SimpleNamespace(
            output_text=(
                "An eval needs `data_source_config` and "
                "`testing_criteria` "
                "[openai-evals-guide-chunk-0004]."
            )
        )

    def parse(
        self,
        *,
        model: str,
        input: list[dict[str, str]],
        text_format: type[JudgeScores],
    ) -> SimpleNamespace:
        self.parse_calls += 1

        return SimpleNamespace(
            output_parsed=JudgeScores(
                relevance=5,
                completeness=4,
                groundedness=5,
                clarity=4,
                reasoning=(
                    "The answer is relevant and supported "
                    "by the retrieved context."
                ),
            )
        )


class FakeOpenAI:
    """Fake OpenAI client for evaluation-runner tests."""

    def __init__(
        self,
        response_error: str | None = None,
    ) -> None:
        self.embeddings = FakeEmbeddings()
        self.responses = FakeResponses(
            error_message=response_error
        )


def create_case() -> BenchmarkCase:
    """Create a benchmark case for runner tests."""

    return BenchmarkCase(
        case_id="eval-0001",
        question="What two ingredients does an eval need?",
        reference_answer=(
            "An eval needs data_source_config "
            "and testing_criteria."
        ),
        category=EvaluationCategory.EVAL_CONCEPTS,
        difficulty=Difficulty.EASY,
        required_facts=[
            "data_source_config",
            "testing_criteria",
        ],
        expected_source_ids=["openai-evals-guide"],
        expected_chunk_ids=[
            "openai-evals-guide-chunk-0004"
        ],
        tags=["evals"],
    )


def create_chunk(
    position: int,
) -> DocumentChunk:
    """Create a document chunk for runner tests."""

    return DocumentChunk(
        chunk_id=f"openai-evals-guide-chunk-{position:04d}",
        source_id="openai-evals-guide",
        position=position,
        heading_path=[
            "Working with evals",
            "Create an eval for a task",
        ],
        content=f"Evaluation guidance number {position}.",
        token_count=5,
    )


def prepare_retrieval_files(
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Create chunk and embedding files for runner tests."""

    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.jsonl"

    chunks = [
        create_chunk(0),
        create_chunk(4),
    ]

    embeddings = [
        ChunkEmbedding(
            chunk_id=chunks[0].chunk_id,
            source_id=chunks[0].source_id,
            embedding_model="text-embedding-3-small",
            embedding=[0.0, 1.0],
        ),
        ChunkEmbedding(
            chunk_id=chunks[1].chunk_id,
            source_id=chunks[1].source_id,
            embedding_model="text-embedding-3-small",
            embedding=[1.0, 0.0],
        ),
    ]

    write_chunks_jsonl(
        chunks=chunks,
        output_path=chunks_path,
    )
    write_embeddings(
        embeddings=embeddings,
        output_path=embeddings_path,
    )

    return chunks_path, embeddings_path


def test_run_evaluation_case_returns_success_record(
    tmp_path: Path,
) -> None:
    chunks_path, embeddings_path = (
        prepare_retrieval_files(tmp_path)
    )
    client = FakeOpenAI()

    record = run_evaluation_case(
        case=create_case(),
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        client=client,
        run_id="test-run-001",
        top_k=1,
    )

    assert record.run_id == "test-run-001"
    assert record.case_id == "eval-0001"
    assert record.status == EvaluationStatus.SUCCESS
    assert record.retrieved_chunk_ids == [
        "openai-evals-guide-chunk-0004"
    ]
    assert len(record.retrieval_scores) == 1
    assert record.answer is not None
    assert record.metrics is not None
    assert record.metrics.overall_score == pytest.approx(1.0)
    assert record.judge_model is None
    assert record.judge_result is None
    assert record.error_message is None
    assert record.duration_ms >= 0.0
    assert client.responses.create_calls == 1
    assert client.responses.parse_calls == 0


def test_run_evaluation_case_records_generation_failure(
    tmp_path: Path,
) -> None:
    chunks_path, embeddings_path = (
        prepare_retrieval_files(tmp_path)
    )

    record = run_evaluation_case(
        case=create_case(),
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        client=FakeOpenAI(
            response_error="Generation service unavailable"
        ),
        run_id="test-run-002",
        top_k=1,
    )

    assert record.status == EvaluationStatus.FAILED
    assert record.answer is None
    assert record.metrics is None
    assert record.judge_result is None
    assert record.error_message == (
        "Generation service unavailable"
    )
    assert record.retrieved_chunk_ids == [
        "openai-evals-guide-chunk-0004"
    ]
    assert record.duration_ms >= 0.0


def test_run_evaluation_case_generates_run_id(
    tmp_path: Path,
) -> None:
    chunks_path, embeddings_path = (
        prepare_retrieval_files(tmp_path)
    )

    record = run_evaluation_case(
        case=create_case(),
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        client=FakeOpenAI(),
        top_k=1,
    )

    assert record.run_id
    assert record.status == EvaluationStatus.SUCCESS


def test_run_evaluation_case_includes_llm_judge_result(
    tmp_path: Path,
) -> None:
    chunks_path, embeddings_path = (
        prepare_retrieval_files(tmp_path)
    )
    client = FakeOpenAI()

    record = run_evaluation_case(
        case=create_case(),
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        client=client,
        run_id="test-run-with-judge",
        top_k=1,
        enable_llm_judge=True,
        judge_model="judge-model",
    )

    assert record.status == EvaluationStatus.SUCCESS
    assert record.judge_model == "judge-model"
    assert record.judge_result is not None
    assert record.judge_result.case_id == "eval-0001"
    assert record.judge_result.relevance == 5
    assert record.judge_result.completeness == 4
    assert record.judge_result.groundedness == 5
    assert record.judge_result.clarity == 4
    assert record.judge_result.overall_score == pytest.approx(
        0.875
    )
    assert record.error_message is None
    assert client.responses.create_calls == 1
    assert client.responses.parse_calls == 1


def test_run_evaluation_case_skips_llm_judge_by_default(
    tmp_path: Path,
) -> None:
    chunks_path, embeddings_path = (
        prepare_retrieval_files(tmp_path)
    )
    client = FakeOpenAI()

    record = run_evaluation_case(
        case=create_case(),
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        client=client,
        run_id="test-run-without-judge",
        top_k=1,
    )

    assert record.status == EvaluationStatus.SUCCESS
    assert record.judge_model is None
    assert record.judge_result is None
    assert client.responses.create_calls == 1
    assert client.responses.parse_calls == 0