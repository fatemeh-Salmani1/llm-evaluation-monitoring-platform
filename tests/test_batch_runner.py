import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.batch_runner import run_benchmark
from src.evaluation.judge import JudgeScores
from src.evaluation.models import (
    BenchmarkCase,
    Difficulty,
    EvaluationCategory,
)
from src.evaluation.runner import EvaluationStatus
from src.retrieval.embedder import ChunkEmbedding
from src.retrieval.embedding_storage import write_embeddings
from src.retrieval.models import DocumentChunk
from src.retrieval.storage import write_chunks_jsonl


class FakeEmbeddings:
    """Fake embeddings endpoint for batch-runner tests."""

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
    """Fake Responses endpoint with configurable failure."""

    def __init__(
        self,
        fail_on_call: int | None = None,
    ) -> None:
        self.fail_on_call = fail_on_call
        self.create_call_count = 0
        self.parse_call_count = 0

    def create(
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        max_output_tokens: int,
    ) -> SimpleNamespace:
        self.create_call_count += 1

        if self.create_call_count == self.fail_on_call:
            raise RuntimeError(
                "Simulated generation failure"
            )

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
        self.parse_call_count += 1

        return SimpleNamespace(
            output_parsed=JudgeScores(
                relevance=5,
                completeness=4,
                groundedness=5,
                clarity=4,
                reasoning=(
                    "The answer is relevant, grounded, "
                    "and mostly complete."
                ),
            )
        )


class FakeOpenAI:
    """Fake OpenAI client for batch-runner tests."""

    def __init__(
        self,
        fail_on_response_call: int | None = None,
    ) -> None:
        self.embeddings = FakeEmbeddings()
        self.responses = FakeResponses(
            fail_on_call=fail_on_response_call
        )


def create_case(
    case_id: str,
) -> BenchmarkCase:
    """Create a benchmark case for batch tests."""

    return BenchmarkCase(
        case_id=case_id,
        question=(
            "What two ingredients does an eval need?"
        ),
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


def prepare_retrieval_files(
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Create stored chunks and embeddings for batch tests."""

    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.jsonl"

    chunk = DocumentChunk(
        chunk_id="openai-evals-guide-chunk-0004",
        source_id="openai-evals-guide",
        position=4,
        heading_path=[
            "Working with evals",
            "Create an eval for a task",
        ],
        content=(
            "An eval needs data_source_config "
            "and testing_criteria."
        ),
        token_count=10,
    )

    embedding = ChunkEmbedding(
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        embedding_model="text-embedding-3-small",
        embedding=[1.0, 0.0],
    )

    write_chunks_jsonl(
        chunks=[chunk],
        output_path=chunks_path,
    )
    write_embeddings(
        embeddings=[embedding],
        output_path=embeddings_path,
    )

    return chunks_path, embeddings_path


def test_run_benchmark_saves_records_and_summary(
    tmp_path: Path,
) -> None:
    chunks_path, embeddings_path = (
        prepare_retrieval_files(tmp_path)
    )
    output_directory = tmp_path / "evaluation-runs"

    result = run_benchmark(
        cases=[
            create_case("eval-0001"),
            create_case("eval-0002"),
        ],
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        output_directory=output_directory,
        client=FakeOpenAI(
            fail_on_response_call=2
        ),
        run_id="test-run-001",
        top_k=1,
    )

    assert len(result.records) == 2
    assert result.records[0].status == (
        EvaluationStatus.SUCCESS
    )
    assert result.records[1].status == (
        EvaluationStatus.FAILED
    )
    assert result.summary.total_cases == 2
    assert result.summary.successful_cases == 1
    assert result.summary.failed_cases == 1
    assert result.summary.success_rate == pytest.approx(0.5)
    assert result.summary.average_overall_score == (
        pytest.approx(1.0)
    )
    assert result.summary.average_judge_score is None
    assert result.summary.judge_model is None
    assert result.summary.failed_case_ids == [
        "eval-0002"
    ]
    assert result.records_path.exists()
    assert result.summary_path.exists()

    record_lines = result.records_path.read_text(
        encoding="utf-8"
    ).splitlines()
    stored_summary = json.loads(
        result.summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert len(record_lines) == 2
    assert stored_summary["run_id"] == "test-run-001"
    assert stored_summary["total_cases"] == 2
    assert stored_summary["average_judge_score"] is None
    assert stored_summary["judge_model"] is None


def test_run_benchmark_aggregates_llm_judge_scores(
    tmp_path: Path,
) -> None:
    chunks_path, embeddings_path = (
        prepare_retrieval_files(tmp_path)
    )
    client = FakeOpenAI()

    result = run_benchmark(
        cases=[
            create_case("eval-0001"),
            create_case("eval-0002"),
        ],
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        output_directory=tmp_path / "evaluation-runs",
        client=client,
        run_id="test-run-with-judge",
        top_k=1,
        enable_llm_judge=True,
        judge_model="judge-model",
    )

    assert result.summary.successful_cases == 2
    assert result.summary.failed_cases == 0
    assert result.summary.judge_model == "judge-model"
    assert result.summary.average_judge_score == (
        pytest.approx(0.875)
    )
    assert all(
        record.judge_result is not None
        for record in result.records
    )
    assert client.responses.create_call_count == 2
    assert client.responses.parse_call_count == 2

    stored_summary = json.loads(
        result.summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert stored_summary["judge_model"] == "judge-model"
    assert stored_summary["average_judge_score"] == (
        pytest.approx(0.875)
    )


def test_run_benchmark_skips_llm_judge_by_default(
    tmp_path: Path,
) -> None:
    chunks_path, embeddings_path = (
        prepare_retrieval_files(tmp_path)
    )
    client = FakeOpenAI()

    result = run_benchmark(
        cases=[create_case("eval-0001")],
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        output_directory=tmp_path / "evaluation-runs",
        client=client,
        run_id="test-run-without-judge",
        top_k=1,
    )

    assert result.summary.average_judge_score is None
    assert result.summary.judge_model is None
    assert result.records[0].judge_result is None
    assert client.responses.parse_call_count == 0


def test_run_benchmark_rejects_empty_judge_model(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="Judge model cannot be empty",
    ):
        run_benchmark(
            cases=[create_case("eval-0001")],
            chunks_path=tmp_path / "chunks.jsonl",
            embeddings_path=tmp_path / "embeddings.jsonl",
            output_directory=tmp_path / "runs",
            client=FakeOpenAI(),
            enable_llm_judge=True,
            judge_model=" ",
        )


def test_run_benchmark_rejects_empty_cases(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least one benchmark case",
    ):
        run_benchmark(
            cases=[],
            chunks_path=tmp_path / "chunks.jsonl",
            embeddings_path=tmp_path / "embeddings.jsonl",
            output_directory=tmp_path / "runs",
            client=FakeOpenAI(),
        )


def test_run_benchmark_rejects_duplicate_case_ids(
    tmp_path: Path,
) -> None:
    case = create_case("eval-0001")

    with pytest.raises(
        ValueError,
        match="unique case IDs",
    ):
        run_benchmark(
            cases=[case, case],
            chunks_path=tmp_path / "chunks.jsonl",
            embeddings_path=tmp_path / "embeddings.jsonl",
            output_directory=tmp_path / "runs",
            client=FakeOpenAI(),
        )


def test_run_benchmark_rejects_unsafe_run_id(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="unsupported characters",
    ):
        run_benchmark(
            cases=[create_case("eval-0001")],
            chunks_path=tmp_path / "chunks.jsonl",
            embeddings_path=tmp_path / "embeddings.jsonl",
            output_directory=tmp_path / "runs",
            client=FakeOpenAI(),
            run_id="../unsafe-run",
        )