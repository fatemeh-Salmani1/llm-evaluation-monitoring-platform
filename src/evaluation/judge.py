from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.evaluation.models import BenchmarkCase
from src.retrieval.models import DocumentChunk

DEFAULT_JUDGE_MODEL = "gpt-5.6-luna"


class JudgeScores(BaseModel):
    """Structured scores returned by the LLM judge."""

    model_config = ConfigDict(frozen=True)

    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    reasoning: str = Field(min_length=1)


class JudgeResult(BaseModel):
    """Validated result produced by the LLM judge."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    judge_model: str
    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    overall_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


class ParsedResponse(Protocol):
    """Response containing parsed structured output."""

    output_parsed: JudgeScores | None


class ResponsesEndpoint(Protocol):
    """Interface required from the Responses API client."""

    def parse(
        self,
        *,
        model: str,
        input: list[dict[str, str]],
        text_format: type[JudgeScores],
    ) -> ParsedResponse:
        """Return a structured response."""


class OpenAIClient(Protocol):
    """Subset of the OpenAI client used by the judge."""

    responses: ResponsesEndpoint


def build_judge_prompt(
    case: BenchmarkCase,
    answer: str,
    retrieved_chunks: list[DocumentChunk],
) -> str:
    """Build the evaluation prompt supplied to the judge."""

    if not answer.strip():
        raise ValueError("Answer cannot be empty")

    if not retrieved_chunks:
        raise ValueError("At least one retrieved chunk is required")

    context = "\n\n".join(
        (
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Heading: {' > '.join(chunk.heading_path)}\n"
            f"Content:\n{chunk.content}"
        )
        for chunk in retrieved_chunks
    )

    required_facts = "\n".join(
        f"- {fact}" for fact in case.required_facts
    )

    return (
        "Evaluate the candidate answer using only the supplied question, "
        "reference answer, required facts, and retrieved context.\n\n"
        "Score each criterion from 1 to 5:\n"
        "- Relevance: directly answers the question.\n"
        "- Completeness: covers the important required facts.\n"
        "- Groundedness: claims are supported by the retrieved context.\n"
        "- Clarity: concise, understandable, and well organized.\n\n"
        "Do not reward information that is unsupported by the context. "
        "Explain the main strengths and weaknesses in the reasoning field."
        "\n\n"
        f"Question:\n{case.question}\n\n"
        f"Reference answer:\n{case.reference_answer}\n\n"
        f"Required facts:\n{required_facts}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"Retrieved context:\n{context}"
    )


def calculate_overall_judge_score(scores: JudgeScores) -> float:
    """Convert the average 1-to-5 judge rating to a 0-to-1 score."""

    total = (
        scores.relevance
        + scores.completeness
        + scores.groundedness
        + scores.clarity
    )
    average = total / 4

    return round((average - 1) / 4, 4)


def judge_answer(
    *,
    case: BenchmarkCase,
    answer: str,
    retrieved_chunks: list[DocumentChunk],
    client: OpenAIClient,
    model: str = DEFAULT_JUDGE_MODEL,
) -> JudgeResult:
    """Evaluate one generated answer with an LLM judge."""

    if not model.strip():
        raise ValueError("Judge model cannot be empty")

    prompt = build_judge_prompt(
        case=case,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
    )

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a strict and consistent evaluator of "
                    "retrieval-augmented LLM answers."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text_format=JudgeScores,
    )

    scores = response.output_parsed

    if scores is None:
        raise ValueError(
            "The judge response did not contain structured scores"
        )

    return JudgeResult(
        case_id=case.case_id,
        judge_model=model,
        relevance=scores.relevance,
        completeness=scores.completeness,
        groundedness=scores.groundedness,
        clarity=scores.clarity,
        overall_score=calculate_overall_judge_score(scores),
        reasoning=scores.reasoning,
    )