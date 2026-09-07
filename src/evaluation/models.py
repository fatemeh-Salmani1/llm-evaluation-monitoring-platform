from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCategory(StrEnum):
    """Technical categories represented in the evaluation benchmark."""

    PYTHON = "python"
    SQL = "sql"
    DATA_MODELING = "data_modeling"
    DATA_QUALITY = "data_quality"
    BATCH_PROCESSING = "batch_processing"
    STREAMING = "streaming"
    CLOUD = "cloud"


class Difficulty(StrEnum):
    """Difficulty assigned to an evaluation case."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BenchmarkCase(BaseModel):
    """A single question and its expected evaluation criteria."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(pattern=r"^de-\d{4}$")
    question: str = Field(min_length=10)
    reference_answer: str = Field(min_length=10)
    category: EvaluationCategory
    difficulty: Difficulty
    required_facts: list[str] = Field(min_length=1)
    expected_source_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)