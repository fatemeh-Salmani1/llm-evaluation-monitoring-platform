import pytest
from pydantic import ValidationError

from src.ingestion.models import DocumentSource


def valid_source_data() -> dict:
    return {
        "source_id": "openai-evals-guide",
        "title": "Working with evals",
        "canonical_url": (
            "https://developers.openai.com/api/docs/guides/evals"
        ),
        "markdown_url": (
            "https://developers.openai.com/api/docs/guides/evals.md"
        ),
        "enabled": True,
    }


def test_document_source_accepts_valid_metadata() -> None:
    source = DocumentSource.model_validate(valid_source_data())

    assert source.source_id == "openai-evals-guide"
    assert source.enabled is True
    assert source.canonical_url.scheme == "https"


def test_document_source_rejects_invalid_source_id() -> None:
    source_data = valid_source_data()
    source_data["source_id"] = "OpenAI Evals Guide"

    with pytest.raises(ValidationError):
        DocumentSource.model_validate(source_data)


def test_document_source_rejects_insecure_url() -> None:
    source_data = valid_source_data()
    source_data["canonical_url"] = (
        "http://developers.openai.com/api/docs/guides/evals"
    )

    with pytest.raises(
        ValidationError,
        match="must use HTTPS",
    ):
        DocumentSource.model_validate(source_data)


def test_document_source_is_immutable() -> None:
    source = DocumentSource.model_validate(valid_source_data())

    with pytest.raises(ValidationError):
        source.title = "Changed title"