BOILERPLATE_PREFIXES = (
    "> For the complete documentation index",
)


def clean_markdown(content: str) -> str:
    """Clean Markdown while preserving headings and code blocks."""

    normalized_content = content.replace("\r\n", "\n").lstrip("\ufeff")
    cleaned_lines: list[str] = []
    inside_code_block = False
    previous_line_blank = False

    for line in normalized_content.splitlines():
        stripped_line = line.strip()

        if stripped_line.startswith("```"):
            inside_code_block = not inside_code_block

        if (
            not inside_code_block
            and stripped_line.startswith(BOILERPLATE_PREFIXES)
        ):
            continue

        current_line_blank = not stripped_line

        if (
            not inside_code_block
            and current_line_blank
            and previous_line_blank
        ):
            continue

        cleaned_lines.append(line.rstrip())

        if not inside_code_block:
            previous_line_blank = current_line_blank

    cleaned_content = "\n".join(cleaned_lines).strip()

    if not cleaned_content:
        raise ValueError("Markdown content is empty after cleaning")

    return f"{cleaned_content}\n"