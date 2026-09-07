import json
from pathlib import Path

from pydantic import ValidationError

from src.evaluation.models import BenchmarkCase


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    """Load and validate benchmark cases from a JSONL file."""

    if not path.exists():
        raise FileNotFoundError(f"Benchmark file does not exist: {path}")

    cases: list[BenchmarkCase] = []
    case_ids: set[str] = set()

    with path.open(encoding="utf-8") as benchmark_file:
        for line_number, line in enumerate(benchmark_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                case = BenchmarkCase.model_validate(record)
            except (json.JSONDecodeError, ValidationError) as error:
                message = (
                    f"Invalid benchmark record in {path} "
                    f"at line {line_number}"
                )
                raise ValueError(message) from error

            if case.case_id in case_ids:
                raise ValueError(
                    f"Duplicate case_id '{case.case_id}' "
                    f"in {path} at line {line_number}"
                )

            case_ids.add(case.case_id)
            cases.append(case)

    return cases