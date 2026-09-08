from pathlib import Path

import httpx
import pytest

from src.ingestion.downloader import download_document
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


def test_download_document_saves_markdown(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/evals.md")

        return httpx.Response(
            status_code=200,
            text="# Working with evals\n\nEvaluation guidance.",
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        output_path = download_document(
            source=create_source(),
            output_directory=tmp_path,
            client=client,
        )

    assert output_path == tmp_path / "openai-evals-guide.md"
    assert output_path.read_text(encoding="utf-8") == (
        "# Working with evals\n\nEvaluation guidance.\n"
    )


def test_download_document_rejects_empty_content(
    tmp_path: Path,
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            status_code=200,
            text="   ",
        )
    )

    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(ValueError, match="empty"),
    ):
        download_document(
            source=create_source(),
            output_directory=tmp_path,
            client=client,
        )


def test_download_document_raises_for_http_error(
    tmp_path: Path,
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            status_code=404,
            request=request,
        )
    )

    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        download_document(
            source=create_source(),
            output_directory=tmp_path,
            client=client,
        )