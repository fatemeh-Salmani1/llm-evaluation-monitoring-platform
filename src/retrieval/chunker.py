import re

from src.retrieval.models import DocumentChunk

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def chunk_markdown(
    source_id: str,
    content: str,
) -> list[DocumentChunk]:
    """Split Markdown into chunks while preserving heading hierarchy."""

    if not content.strip():
        raise ValueError("Cannot chunk empty Markdown content")

    chunks: list[DocumentChunk] = []
    heading_path: list[str] = []
    current_lines: list[str] = []
    inside_code_block = False

    def save_current_chunk() -> None:
        chunk_content = "\n".join(current_lines).strip()

        if not chunk_content:
            return

        position = len(chunks)
        chunks.append(
            DocumentChunk(
                chunk_id=f"{source_id}-chunk-{position:04d}",
                source_id=source_id,
                position=position,
                heading_path=heading_path.copy() or ["Untitled"],
                content=chunk_content,
            )
        )

    for line in content.splitlines():
        stripped_line = line.strip()

        if stripped_line.startswith("```"):
            inside_code_block = not inside_code_block

        heading_match = None

        if not inside_code_block:
            heading_match = HEADING_PATTERN.match(line)

        if heading_match:
            save_current_chunk()
            current_lines.clear()

            heading_level = len(heading_match.group(1))
            heading_title = heading_match.group(2).strip()
            heading_path = (
                heading_path[: heading_level - 1]
                + [heading_title]
            )

        current_lines.append(line)

    save_current_chunk()

    return chunks