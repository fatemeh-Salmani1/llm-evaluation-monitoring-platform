import json
from pathlib import Path

import httpx

from src.ingestion.run import ingest_sources


def test_ingest_sources_processes_only_enabled_sources(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "sources.json"
    raw_directory = tmp_path / "raw"
    processed_directory = tmp_path / "processed"

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
            text=(
                "# Downloaded guide\n\n"
                "> For the complete documentation index, "
                "see [llms.txt](/llms.txt).\n\n\n"
                "Evaluation guidance."
            ),
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        processed_paths = ingest_sources(
            manifest_path=manifest_path,
            raw_output_directory=raw_directory,
            processed_output_directory=processed_directory,
            client=client,
        )

    raw_path = raw_directory / "enabled-guide.md"
    metadata_path = (
        raw_directory / "enabled-guide.metadata.json"
    )
    processed_path = processed_directory / "enabled-guide.md"

    assert requested_paths == ["/enabled.md"]
    assert processed_paths == [processed_path]

    assert raw_path.exists()
    assert metadata_path.exists()
    assert processed_path.exists()

    raw_content = raw_path.read_text(encoding="utf-8")
    processed_content = processed_path.read_text(encoding="utf-8")

    assert "complete documentation index" in raw_content
    assert "complete documentation index" not in processed_content
    assert processed_content.endswith("Evaluation guidance.\n")

    saved_metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert saved_metadata["source_id"] == "enabled-guide"
    assert saved_metadata["content_bytes"] > 0
    assert len(saved_metadata["content_sha256"]) == 64

    assert not (raw_directory / "disabled-guide.md").exists()
    assert not (
        processed_directory / "disabled-guide.md"
    ).exists()