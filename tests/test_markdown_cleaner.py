import pytest

from src.ingestion.cleaner import clean_markdown


def test_clean_markdown_removes_documentation_boilerplate() -> None:
    content = (
        "# Working with evals\n\n"
        "> For the complete documentation index, "
        "see [llms.txt](/llms.txt).\n\n"
        "Evaluations test model outputs.\n"
    )

    cleaned_content = clean_markdown(content)

    assert cleaned_content.startswith("# Working with evals")
    assert "complete documentation index" not in cleaned_content
    assert "Evaluations test model outputs." in cleaned_content


def test_clean_markdown_collapses_extra_blank_lines() -> None:
    content = "# Guide\n\n\n\nFirst paragraph.\n\n\nSecond paragraph."

    cleaned_content = clean_markdown(content)

    assert cleaned_content == (
        "# Guide\n\n"
        "First paragraph.\n\n"
        "Second paragraph.\n"
    )


def test_clean_markdown_preserves_fenced_code_blocks() -> None:
    content = (
        "# Example\n\n"
        "```python\n"
        "def evaluate():\n\n"
        "    return True\n"
        "```\n"
    )

    cleaned_content = clean_markdown(content)

    assert "def evaluate():\n\n    return True" in cleaned_content
    assert "```python" in cleaned_content


def test_clean_markdown_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="empty after cleaning"):
        clean_markdown("   \n\n   ")