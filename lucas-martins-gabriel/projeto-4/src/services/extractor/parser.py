from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter


def parse_pdf_to_markdown(pdf_path: str | Path) -> str:
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def write_markdown(markdown: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path
