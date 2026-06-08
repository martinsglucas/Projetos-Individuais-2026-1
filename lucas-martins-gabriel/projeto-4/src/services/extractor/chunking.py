from __future__ import annotations

from collections.abc import Iterable

from contracts import DocumentChunk


def estimate_token_count(text: str) -> int:
    return len(text.split())


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("#") and stripped.lstrip("#").strip() != ""


def _clean_lines(markdown: str) -> list[str]:
    lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped == "<!-- image -->":
            continue
        lines.append(line.rstrip())
    return lines


def _split_large_block(lines: Iterable[str], max_chars: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    current_size = 0

    for line in lines:
        additional = len(line) + 1
        if current and current_size + additional > max_chars:
            parts.append("\n".join(current).strip())
            current = []
            current_size = 0
        current.append(line)
        current_size += additional

    if current:
        parts.append("\n".join(current).strip())

    return [part for part in parts if part]


def chunk_markdown(markdown: str, document_hash: str, max_chars: int = 6000) -> list[DocumentChunk]:
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in _clean_lines(markdown):
        if _is_heading(line):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    if not sections and markdown.strip():
        sections.append((None, markdown.splitlines()))

    chunks: list[DocumentChunk] = []
    ordinal = 0
    for heading, lines in sections:
        for content in _split_large_block(lines, max_chars=max_chars):
            token_count = estimate_token_count(content)
            if token_count < 3:
                continue
            chunks.append(
                DocumentChunk(
                    document_hash=document_hash,
                    ordinal=ordinal,
                    heading=heading,
                    content=content,
                    token_count=token_count,
                    parser="docling",
                    metadata={"chunking_strategy": "markdown_heading", "max_chars": max_chars},
                )
            )
            ordinal += 1

    return chunks
