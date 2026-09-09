import json
from pathlib import Path

import pytest

from src.evaluation.loader import load_benchmark


def valid_record(
    case_id: str = "eval-0001",
) -> dict[str, object]:
    """Create a valid benchmark record for loader tests."""

    return {
        "case_id": case_id,
        "question": (
            "What are the two key ingredients of an eval?"
        ),
        "reference_answer": (
            "An eval needs a data source configuration "
            "and testing criteria."
        ),
        "category": "eval_concepts",
        "difficulty": "easy",
        "required_facts": [
            "An eval needs data_source_config.",
            "An eval needs testing_criteria.",
        ],
        "expected_source_ids": [
            "openai-evals-guide"
        ],
        "expected_chunk_ids": [
            "openai-evals-guide-chunk-0004"
        ],
        "tags": [
            "evals",
            "configuration",
            "graders",
        ],
    }


def write_jsonl(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    """Write benchmark records as JSONL."""

    content = "\n".join(
        json.dumps(record)
        for record in records
    )
    path.write_text(
        content + "\n",
        encoding="utf-8",
    )


def test_load_benchmark_returns_validated_cases(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"

    write_jsonl(
        benchmark_path,
        [
            valid_record("eval-0001"),
            valid_record("eval-0002"),
        ],
    )

    cases = load_benchmark(benchmark_path)

    assert len(cases) == 2
    assert cases[0].case_id == "eval-0001"
    assert cases[1].case_id == "eval-0002"
    assert cases[0].expected_chunk_ids == [
        "openai-evals-guide-chunk-0004"
    ]


def test_load_benchmark_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text(
        "{invalid-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="line 1",
    ):
        load_benchmark(benchmark_path)


def test_load_benchmark_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"

    write_jsonl(
        benchmark_path,
        [
            valid_record("eval-0001"),
            valid_record("eval-0001"),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate case_id",
    ):
        load_benchmark(benchmark_path)


def test_load_benchmark_rejects_missing_file(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError):
        load_benchmark(benchmark_path)