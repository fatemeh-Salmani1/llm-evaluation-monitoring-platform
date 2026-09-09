import re

from tiktoken import Encoding, get_encoding

from src.retrieval.models import DocumentChunk

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DEFAULT_MAX_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 75
DEFAULT_ENCODING = "cl100k_base"


def extract_markdown_sections(
    content: str,
) -> list[tuple[list[str], str]]:
    """Split Markdown into sections using its heading hierarchy."""

    sections: list[tuple[list[str], str]] = []
    heading_path: list[str] = []
    current_lines: list[str] = []
    inside_code_block = False

    def save_current_section() -> None:
        section_content = "\n".join(current_lines).strip()

        if section_content:
            sections.append(
                (
                    heading_path.copy() or ["Untitled"],
                    section_content,
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
            save_current_section()
            current_lines.clear()

            heading_level = len(heading_match.group(1))
            heading_title = heading_match.group(2).strip()
            heading_path = (
                heading_path[: heading_level - 1]
                + [heading_title]
            )

        current_lines.append(line)

    save_current_section()

    return sections


def split_section_by_tokens(
    content: str,
    encoding: Encoding,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, int]]:
    """Split one section into overlapping token-limited pieces."""

    token_ids = encoding.encode(content)
    pieces: list[tuple[str, int]] = []
    start = 0

    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        piece_content = encoding.decode(token_ids[start:end])

        if piece_content.strip():
            piece_token_count = end - start
            pieces.append((piece_content, piece_token_count))

        if end == len(token_ids):
            break

        start = end - overlap_tokens

    return pieces


def chunk_markdown(
    source_id: str,
    content: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[DocumentChunk]:
    """Create heading-aware, token-limited Markdown chunks."""

    if not content.strip():
        raise ValueError("Cannot chunk empty Markdown content")

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError(
            "overlap_tokens must be between zero and max_tokens"
        )

    encoding = get_encoding(encoding_name)
    sections = extract_markdown_sections(content)
    chunks: list[DocumentChunk] = []

    for heading_path, section_content in sections:
        pieces = split_section_by_tokens(
            content=section_content,
            encoding=encoding,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        for piece_content, token_count in pieces:
            position = len(chunks)
            chunks.append(
                DocumentChunk(
                    chunk_id=(
                        f"{source_id}-chunk-{position:04d}"
                    ),
                    source_id=source_id,
                    position=position,
                    heading_path=heading_path,
                    content=piece_content,
                    token_count=token_count,
                )
            )

    return chunks