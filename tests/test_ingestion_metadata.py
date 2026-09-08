import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.ingestion.metadata import (
    build_document_metadata,
    calculate_sha256,
    write_document_metadata,
)
from src.ingestion.models import DocumentSource


def create_source() -> DocumentSource:
    return DocumentSource(
        source_id="openai-evals-guide",
        title="Working with evals",
        canonical_url=(
            "https://developers.openai.com/api/docs/guides/evals"
        ),
        markdown_url=(
            "https://developers.openai.com/api/docs/guides/evals.md"
        ),
    )


def test_calculate_sha256_returns_expected_checksum(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "guide.md"
    content = b"# Evaluation guide\n"
    document_path.write_bytes(content)

    checksum = calculate_sha256(document_path)

    assert checksum == hashlib.sha256(content).hexdigest()


def test_build_document_metadata_records_lineage(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "guide.md"
    document_path.write_text("# Evaluation guide\n", encoding="utf-8")
    retrieved_at = datetime(2026, 9, 8, 18, 30, tzinfo=UTC)

    metadata = build_document_metadata(
        source=create_source(),
        document_path=document_path,
        retrieved_at=retrieved_at,
    )

    assert metadata.source_id == "openai-evals-guide"
    assert metadata.local_path == document_path.as_posix()
    assert metadata.content_bytes == document_path.stat().st_size
    assert metadata.retrieved_at == retrieved_at
    assert len(metadata.content_sha256) == 64


def test_calculate_sha256_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        calculate_sha256(tmp_path / "missing.md")


def test_write_document_metadata_saves_json(
    tmp_path: Path,
) -> None:
    document_path = tmp_path / "guide.md"
    document_path.write_text("# Evaluation guide\n", encoding="utf-8")

    metadata = build_document_metadata(
        source=create_source(),
        document_path=document_path,
        retrieved_at=datetime(2026, 9, 8, 18, 30, tzinfo=UTC),
    )
    metadata_path = tmp_path / "guide.metadata.json"

    returned_path = write_document_metadata(
        metadata=metadata,
        output_path=metadata_path,
    )
    saved_metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert returned_path == metadata_path
    assert saved_metadata["source_id"] == "openai-evals-guide"
    assert saved_metadata["content_bytes"] > 0
    assert len(saved_metadata["content_sha256"]) == 64