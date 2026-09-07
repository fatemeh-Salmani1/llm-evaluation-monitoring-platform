import json
from pathlib import Path

import pytest

from src.evaluation.loader import load_benchmark


def valid_record(case_id: str = "de-0001") -> dict:
    return {
        "case_id": case_id,
        "question": "What is the purpose of a primary key?",
        "reference_answer": (
            "A primary key uniquely identifies each row in a table."
        ),
        "category": "sql",
        "difficulty": "easy",
        "required_facts": [
            "A primary key uniquely identifies each row.",
        ],
        "expected_source_ids": ["database-keys"],
        "tags": ["sql", "primary-key"],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    content = "\n".join(json.dumps(record) for record in records)
    path.write_text(content, encoding="utf-8")


def test_load_benchmark_returns_validated_cases(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"
    write_jsonl(
        benchmark_path,
        [
            valid_record("de-0001"),
            valid_record("de-0002"),
        ],
    )

    cases = load_benchmark(benchmark_path)

    assert len(cases) == 2
    assert cases[0].case_id == "de-0001"
    assert cases[1].case_id == "de-0002"


def test_load_benchmark_rejects_invalid_json(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text("{invalid-json}", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_benchmark(benchmark_path)


def test_load_benchmark_rejects_duplicate_ids(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"
    write_jsonl(
        benchmark_path,
        [
            valid_record("de-0001"),
            valid_record("de-0001"),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate case_id"):
        load_benchmark(benchmark_path)


def test_load_benchmark_rejects_missing_file(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError):
        load_benchmark(benchmark_path)