import pytest

from src.retrieval.chunker import chunk_markdown


def test_chunk_markdown_splits_content_by_headings() -> None:
    content = (
        "# Evaluation guide\n\n"
        "Introduction.\n\n"
        "## Create an evaluation\n\n"
        "Creation guidance.\n\n"
        "### Add test data\n\n"
        "Dataset guidance."
    )

    chunks = chunk_markdown(
        source_id="openai-evals-guide",
        content=content,
    )

    assert len(chunks) == 3
    assert chunks[0].heading_path == ["Evaluation guide"]
    assert chunks[1].heading_path == [
        "Evaluation guide",
        "Create an evaluation",
    ]
    assert chunks[2].heading_path == [
        "Evaluation guide",
        "Create an evaluation",
        "Add test data",
    ]


def test_chunk_markdown_assigns_deterministic_ids() -> None:
    content = "# Guide\n\nIntroduction.\n\n## Dataset\n\nDetails."

    chunks = chunk_markdown(
        source_id="openai-evals-guide",
        content=content,
    )

    assert chunks[0].chunk_id == (
        "openai-evals-guide-chunk-0000"
    )
    assert chunks[1].chunk_id == (
        "openai-evals-guide-chunk-0001"
    )
    assert chunks[0].position == 0
    assert chunks[1].position == 1


def test_chunk_markdown_ignores_headings_inside_code_blocks() -> None:
    content = (
        "# Guide\n\n"
        "```markdown\n"
        "# This is code, not a document heading\n"
        "```\n\n"
        "Final explanation."
    )

    chunks = chunk_markdown(
        source_id="openai-evals-guide",
        content=content,
    )

    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Guide"]
    assert "# This is code" in chunks[0].content


def test_chunk_markdown_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="empty Markdown"):
        chunk_markdown(
            source_id="openai-evals-guide",
            content="   \n\n",
        )