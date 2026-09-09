import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.evaluation.models import BenchmarkCase

CHUNK_CITATION_PATTERN = re.compile(
    r"\[([a-z0-9]+(?:-[a-z0-9]+)*-chunk-\d{4})\]"
)


class DeterministicEvaluationResult(BaseModel):
    """Deterministic quality metrics for one benchmark case."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    retrieval_hit: bool
    retrieval_recall: float = Field(ge=0.0, le=1.0)
    fact_coverage: float = Field(ge=0.0, le=1.0)
    citation_validity: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    matched_facts: list[str]
    missing_facts: list[str]
    cited_chunk_ids: list[str]
    invalid_citation_ids: list[str]


def normalize_text(text: str) -> str:
    """Normalize text for deterministic phrase matching."""

    normalized = re.sub(
        r"[^a-z0-9_]+",
        " ",
        text.lower(),
    )

    return " ".join(normalized.split())


def extract_chunk_citations(answer: str) -> list[str]:
    """Extract unique chunk citations while preserving their order."""

    citations = CHUNK_CITATION_PATTERN.findall(answer)
    unique_citations: list[str] = []
    seen: set[str] = set()

    for citation in citations:
        if citation not in seen:
            unique_citations.append(citation)
            seen.add(citation)

    return unique_citations


def calculate_retrieval_metrics(
    expected_chunk_ids: Sequence[str],
    retrieved_chunk_ids: Sequence[str],
) -> tuple[bool, float]:
    """Calculate retrieval hit and recall at the selected K."""

    if not expected_chunk_ids:
        return False, 0.0

    expected_ids = set(expected_chunk_ids)
    retrieved_ids = set(retrieved_chunk_ids)
    matched_ids = expected_ids & retrieved_ids

    retrieval_hit = bool(matched_ids)
    retrieval_recall = len(matched_ids) / len(expected_ids)

    return retrieval_hit, retrieval_recall


def calculate_fact_coverage(
    required_facts: Sequence[str],
    answer: str,
) -> tuple[float, list[str], list[str]]:
    """Measure how many required fact phrases appear in an answer."""

    if not required_facts:
        raise ValueError("At least one required fact is needed")

    normalized_answer = normalize_text(answer)
    matched_facts: list[str] = []
    missing_facts: list[str] = []

    for fact in required_facts:
        normalized_fact = normalize_text(fact)

        if normalized_fact in normalized_answer:
            matched_facts.append(fact)
        else:
            missing_facts.append(fact)

    coverage = len(matched_facts) / len(required_facts)

    return coverage, matched_facts, missing_facts


def calculate_citation_validity(
    answer: str,
    retrieved_chunk_ids: Sequence[str],
) -> tuple[float, list[str], list[str]]:
    """Measure whether answer citations refer to retrieved chunks."""

    cited_chunk_ids = extract_chunk_citations(answer)

    if not cited_chunk_ids:
        return 0.0, [], []

    retrieved_ids = set(retrieved_chunk_ids)
    invalid_citation_ids = [
        citation
        for citation in cited_chunk_ids
        if citation not in retrieved_ids
    ]
    valid_count = (
        len(cited_chunk_ids)
        - len(invalid_citation_ids)
    )
    validity = valid_count / len(cited_chunk_ids)

    return (
        validity,
        cited_chunk_ids,
        invalid_citation_ids,
    )


def evaluate_deterministically(
    case: BenchmarkCase,
    answer: str,
    retrieved_chunk_ids: Sequence[str],
) -> DeterministicEvaluationResult:
    """Evaluate retrieval, facts, and citations for one case."""

    if not answer.strip():
        raise ValueError("Answer cannot be empty")

    retrieval_hit, retrieval_recall = (
        calculate_retrieval_metrics(
            expected_chunk_ids=case.expected_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )
    )
    fact_coverage, matched_facts, missing_facts = (
        calculate_fact_coverage(
            required_facts=case.required_facts,
            answer=answer,
        )
    )
    (
        citation_validity,
        cited_chunk_ids,
        invalid_citation_ids,
    ) = calculate_citation_validity(
        answer=answer,
        retrieved_chunk_ids=retrieved_chunk_ids,
    )

    overall_score = (
        float(retrieval_hit)
        + retrieval_recall
        + fact_coverage
        + citation_validity
    ) / 4

    return DeterministicEvaluationResult(
        case_id=case.case_id,
        retrieval_hit=retrieval_hit,
        retrieval_recall=retrieval_recall,
        fact_coverage=fact_coverage,
        citation_validity=citation_validity,
        overall_score=overall_score,
        matched_facts=matched_facts,
        missing_facts=missing_facts,
        cited_chunk_ids=cited_chunk_ids,
        invalid_citation_ids=invalid_citation_ids,
    )