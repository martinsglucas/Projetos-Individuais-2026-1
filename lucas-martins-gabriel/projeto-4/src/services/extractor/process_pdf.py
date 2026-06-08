from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from contracts import CompanyCode, DocumentMetadata, Period, ReportType
from db import UdaRepository
from services.extractor.chunking import chunk_markdown
from services.extractor.hash import calculate_sha256
from services.extractor.parser import parse_pdf_to_markdown, write_markdown
from services.storage import artifact_name, get_artifact_storage

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
DEFAULT_PARSED_DIR = SRC_ROOT / "data" / "parsed"


def load_dotenv_file(path: Path = SRC_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def resolve_input_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.resolve()

    src_candidate = SRC_ROOT / value
    if src_candidate.exists():
        return src_candidate.resolve()

    project_candidate = PROJECT_ROOT / value
    if project_candidate.exists():
        return project_candidate.resolve()

    raise FileNotFoundError(f"PDF not found: {value}")


def path_for_display(path: Path) -> str:
    try:
        return str(path.relative_to(SRC_ROOT))
    except ValueError:
        return str(path)


def parsed_output_path(company: CompanyCode, period: Period, pdf_hash: str, output_dir: Path) -> Path:
    filename = f"{company.value.lower()}_{period.label.lower()}_{pdf_hash[:12]}.md"
    return output_dir / filename


def process_pdf(
    pdf_path: Path,
    company: CompanyCode,
    period: Period,
    source_url: str | None,
    report_type: ReportType,
    parsed_dir: Path = DEFAULT_PARSED_DIR,
) -> str:
    repo = UdaRepository()
    storage = get_artifact_storage()
    pdf_hash = calculate_sha256(pdf_path)
    now = datetime.utcnow()
    raw_storage_path = storage.put_file(
        path=pdf_path,
        object_name=artifact_name("raw", company.value, str(period.year), period.label, f"{pdf_hash}.pdf"),
        content_type="application/pdf",
    )

    document = DocumentMetadata(
        company=company,
        report_type=report_type,
        period=period,
        source_url=source_url,
        pdf_hash=pdf_hash,
        storage_path=raw_storage_path,
        original_filename=pdf_path.name,
        discovered_at=now,
        downloaded_at=now,
    )

    document_id = repo.upsert_document(document, status="downloaded")
    markdown = parse_pdf_to_markdown(pdf_path)

    output_path = parsed_output_path(company, period, pdf_hash, parsed_dir)
    write_markdown(markdown, output_path)
    parsed_storage_path = storage.put_text(
        text=markdown,
        object_name=artifact_name("parsed", company.value, str(period.year), period.label, f"{pdf_hash}.md"),
        content_type="text/markdown; charset=utf-8",
    )

    chunks = chunk_markdown(markdown, document_hash=pdf_hash)
    chunk_ids = repo.replace_chunks(pdf_hash, chunks)
    repo.mark_document_parsed(pdf_hash, parsed_storage_path=parsed_storage_path)

    print(f"document_id={document_id}")
    print(f"pdf_hash={pdf_hash}")
    print(f"parsed_markdown={path_for_display(output_path)}")
    print(f"raw_storage={raw_storage_path}")
    print(f"parsed_storage={parsed_storage_path}")
    print(f"chunks={len(chunk_ids)}")
    return pdf_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a PDF into markdown chunks and persist them in Postgres.")
    parser.add_argument("--pdf", required=True, help="Path to the local PDF file.")
    parser.add_argument("--company", required=True, choices=[item.value for item in CompanyCode])
    parser.add_argument("--period", required=True, help="Quarter label, e.g. 1T25.")
    parser.add_argument("--source-url", default=None, help="Original RI URL for lineage.")
    parser.add_argument(
        "--report-type",
        default=ReportType.OPERATIONAL_PREVIEW.value,
        choices=[item.value for item in ReportType],
    )
    parser.add_argument("--parsed-dir", default=str(DEFAULT_PARSED_DIR), help="Directory for parsed markdown artifacts.")
    return parser.parse_args()


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    process_pdf(
        pdf_path=resolve_input_path(args.pdf),
        company=CompanyCode(args.company),
        period=Period.from_label(args.period),
        source_url=args.source_url,
        report_type=ReportType(args.report_type),
        parsed_dir=Path(args.parsed_dir),
    )


if __name__ == "__main__":
    main()
