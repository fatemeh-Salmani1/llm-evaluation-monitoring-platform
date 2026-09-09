from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCategory(StrEnum):
    """Categories represented in the LLM evaluation benchmark."""

    EVAL_CONCEPTS = "eval_concepts"
    DATA_SOURCES = "data_sources"
    GRADERS = "graders"
    EVAL_RUNS = "eval_runs"
    API_USAGE = "api_usage"
    BEST_PRACTICES = "best_practices"


class Difficulty(StrEnum):
    """Difficulty assigned to an evaluation case."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BenchmarkCase(BaseModel):
    """A question and its expected evaluation criteria."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(pattern=r"^eval-\d{4}$")
    question: str = Field(min_length=10)
    reference_answer: str = Field(min_length=10)
    category: EvaluationCategory
    difficulty: Difficulty
    required_facts: list[str] = Field(min_length=1)
    expected_source_ids: list[str] = Field(
        default_factory=list
    )
    expected_chunk_ids: list[str] = Field(
        default_factory=list
    )
    tags: list[str] = Field(default_factory=list)