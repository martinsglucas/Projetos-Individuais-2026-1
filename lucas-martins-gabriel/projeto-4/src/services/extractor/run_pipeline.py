from __future__ import annotations

import argparse
from pathlib import Path

from contracts import CompanyCode, Period, ReportType
from db import UdaRepository
from services.extractor.extract_metrics import run_extraction
from services.extractor.hash import calculate_sha256
from services.extractor.process_pdf import DEFAULT_PARSED_DIR, load_dotenv_file, process_pdf, resolve_input_path


def run_pipeline(
    *,
    pdf_path: Path,
    company: CompanyCode,
    period: Period,
    source_url: str | None,
    report_type: ReportType,
    model_name: str,
    fixture_path: Path | None = None,
    force: bool = False,
) -> None:
    repo = UdaRepository()
    pdf_hash = calculate_sha256(pdf_path)

    if repo.document_exists_by_hash(pdf_hash) and not force:
        document = repo.get_document_by_hash(pdf_hash)
        if document["status"] in {"extracted", "validated"}:
            print(f"skip reason=already_processed hash={pdf_hash} status={document['status']}")
            return
        if document["status"] == "parsed":
            print(f"reuse_existing_chunks hash={pdf_hash} status={document['status']}")
        else:
            pdf_hash = process_pdf(
                pdf_path=pdf_path,
                company=company,
                period=period,
                source_url=source_url,
                report_type=report_type,
                parsed_dir=DEFAULT_PARSED_DIR,
            )
    else:
        pdf_hash = process_pdf(
            pdf_path=pdf_path,
            company=company,
            period=period,
            source_url=source_url,
            report_type=report_type,
            parsed_dir=DEFAULT_PARSED_DIR,
        )

    run_extraction(
        document_hash=pdf_hash,
        model_name=model_name,
        fixture_path=fixture_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local PDF UDA pipeline end to end.")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--company", required=True, choices=[item.value for item in CompanyCode])
    parser.add_argument("--period", required=True, help="Quarter label, e.g. 1T25.")
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--fixture", default=None, help="JSON fixture for offline extraction tests.")
    parser.add_argument("--force", action="store_true", help="Reprocess even when the hash already exists.")
    parser.add_argument(
        "--report-type",
        default=ReportType.OPERATIONAL_PREVIEW.value,
        choices=[item.value for item in ReportType],
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    run_pipeline(
        pdf_path=resolve_input_path(args.pdf),
        company=CompanyCode(args.company),
        period=Period.from_label(args.period),
        source_url=args.source_url,
        report_type=ReportType(args.report_type),
        model_name=args.model,
        fixture_path=Path(args.fixture).resolve() if args.fixture else None,
        force=args.force,
    )


if __name__ == "__main__":
    main()
