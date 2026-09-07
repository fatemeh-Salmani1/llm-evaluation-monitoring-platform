import json
from pathlib import Path

import pytest

from src.ingestion.source_loader import load_document_sources

PROJECT_ROOT = Path(__file__).parents[1]


def valid_source(source_id: str = "openai-evals-guide") -> dict:
    return {
        "source_id": source_id,
        "title": "Working with evals",
        "canonical_url": (
            "https://developers.openai.com/api/docs/guides/evals"
        ),
        "markdown_url": (
            "https://developers.openai.com/api/docs/guides/evals.md"
        ),
        "enabled": True,
    }


def test_project_source_manifest_is_valid() -> None:
    manifest_path = (
        PROJECT_ROOT / "data" / "sources" / "openai_docs.json"
    )

    sources = load_document_sources(manifest_path)

    assert len(sources) == 1
    assert sources[0].source_id == "openai-evals-guide"


def test_source_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(
        json.dumps(
            [
                valid_source(),
                valid_source(),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate source IDs"):
        load_document_sources(manifest_path)


def test_source_loader_rejects_non_list_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(
        json.dumps(valid_source()),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="JSON array"):
        load_document_sources(manifest_path)


def test_source_loader_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_document_sources(tmp_path / "missing.json")