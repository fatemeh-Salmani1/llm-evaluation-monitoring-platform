from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.batch_runner import (
    BatchEvaluationSummary,
)
from src.evaluation.judge import JudgeResult
from src.evaluation.models import (
    Difficulty,
    EvaluationCategory,
)
from src.evaluation.runner import (
    EvaluationRunRecord,
    EvaluationStatus,
)
from src.monitoring.dashboard_data import (
    build_case_frame,
    build_history_frame,
    discover_summary_paths,
    load_evaluation_records,
    load_run_history,
    records_path_for_run,
)


def create_summary(
    run_id: str,
    minute: int,
    judge_score: float | None = None,
) -> BatchEvaluationSummary:
    """Create a benchmark summary for dashboard tests."""

    started_at = datetime(
        2026,
        9,
        10,
        10,
        minute,
        tzinfo=UTC,
    )

    return BatchEvaluationSummary(
        run_id=run_id,
        started_at=started_at,
        completed_at=started_at,
        total_cases=8,
        successful_cases=8,
        failed_cases=0,
        success_rate=1.0,
        average_duration_ms=2000.0,
        average_retrieval_recall=1.0,
        average_fact_coverage=0.9,
        average_citation_validity=1.0,
        average_overall_score=0.975,
        average_judge_score=judge_score,
        failed_case_ids=[],
        embedding_model="text-embedding-3-small",
        generation_model="gpt-5.6-luna",
        judge_model=(
            "gpt-5.6-luna"
            if judge_score is not None
            else None
        ),
        top_k=4,
    )


def create_record() -> EvaluationRunRecord:
    """Create a case record containing an LLM judge result."""

    return EvaluationRunRecord(
        run_id="run-002",
        case_id="eval-0001",
        status=EvaluationStatus.SUCCESS,
        started_at=datetime(
            2026,
            9,
            10,
            10,
            1,
            tzinfo=UTC,
        ),
        duration_ms=2500.0,
        question="What is an LLM evaluation?",
        category=EvaluationCategory.EVAL_CONCEPTS,
        difficulty=Difficulty.EASY,
        embedding_model="text-embedding-3-small",
        generation_model="gpt-5.6-luna",
        judge_model="gpt-5.6-luna",
        retrieved_chunk_ids=[
            "openai-evals-guide-chunk-0000"
        ],
        retrieval_scores=[0.8],
        answer="An evaluation tests model outputs.",
        judge_result=JudgeResult(
            case_id="eval-0001",
            judge_model="gpt-5.6-luna",
            relevance=5,
            completeness=4,
            groundedness=5,
            clarity=5,
            overall_score=0.9375,
            reasoning="The answer is grounded but slightly brief.",
        ),
    )


def test_discover_summary_paths_ignores_other_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "run-001.summary.json"
    second = tmp_path / "run-002.summary.json"

    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    (tmp_path / "run-001.jsonl").write_text(
        "{}",
        encoding="utf-8",
    )

    paths = discover_summary_paths(tmp_path)

    assert paths == [first, second]


def test_load_run_history_orders_runs_by_completion(
    tmp_path: Path,
) -> None:
    later = create_summary(
        run_id="later-run",
        minute=2,
    )
    earlier = create_summary(
        run_id="earlier-run",
        minute=1,
    )

    (tmp_path / "a.summary.json").write_text(
        later.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (tmp_path / "b.summary.json").write_text(
        earlier.model_dump_json(indent=2),
        encoding="utf-8",
    )

    summaries = load_run_history(tmp_path)

    assert [
        summary.run_id
        for summary in summaries
    ] == [
        "earlier-run",
        "later-run",
    ]


def test_load_run_history_returns_empty_for_missing_directory(
    tmp_path: Path,
) -> None:
    summaries = load_run_history(
        tmp_path / "missing"
    )

    assert summaries == []


def test_build_history_frame_includes_judge_score() -> None:
    frame = build_history_frame(
        [
            create_summary(
                run_id="run-001",
                minute=1,
                judge_score=0.95,
            )
        ]
    )

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert frame.loc[0, "run_id"] == "run-001"
    assert frame.loc[0, "judge_score"] == pytest.approx(
        0.95
    )


def test_load_evaluation_records_round_trip(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "run-002.jsonl"
    record = create_record()

    records_path.write_text(
        record.model_dump_json() + "\n",
        encoding="utf-8",
    )

    loaded = load_evaluation_records(records_path)

    assert loaded == [record]


def test_build_case_frame_includes_judge_dimensions() -> None:
    frame = build_case_frame([create_record()])

    assert len(frame) == 1
    assert frame.loc[0, "case_id"] == "eval-0001"
    assert frame.loc[0, "judge_score"] == pytest.approx(
        0.9375
    )
    assert frame.loc[0, "judge_relevance"] == 5
    assert frame.loc[0, "judge_completeness"] == 4
    assert (
        frame.loc[0, "judge_reasoning"]
        == "The answer is grounded but slightly brief."
    )


def test_records_path_for_run_rejects_unsafe_id(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="unsupported path characters",
    ):
        records_path_for_run(
            output_directory=tmp_path,
            run_id="../unsafe",
        )