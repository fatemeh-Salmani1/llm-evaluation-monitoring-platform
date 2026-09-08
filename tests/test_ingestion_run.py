import json
from pathlib import Path

import httpx

from src.ingestion.run import ingest_sources


def test_ingest_sources_downloads_only_enabled_sources(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "sources.json"
    output_directory = tmp_path / "downloads"

    manifest_path.write_text(
        json.dumps(
            [
                {
                    "source_id": "enabled-guide",
                    "title": "Enabled guide",
                    "canonical_url": "https://example.com/enabled",
                    "markdown_url": "https://example.com/enabled.md",
                    "enabled": True,
                },
                {
                    "source_id": "disabled-guide",
                    "title": "Disabled guide",
                    "canonical_url": "https://example.com/disabled",
                    "markdown_url": "https://example.com/disabled.md",
                    "enabled": False,
                },
            ]
        ),
        encoding="utf-8",
    )

    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)

        return httpx.Response(
            status_code=200,
            text="# Downloaded guide",
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        downloaded_paths = ingest_sources(
            manifest_path=manifest_path,
            output_directory=output_directory,
            client=client,
        )

    assert requested_paths == ["/enabled.md"]
    assert downloaded_paths == [
        output_directory / "enabled-guide.md"
    ]
    assert not (output_directory / "disabled-guide.md").exists()