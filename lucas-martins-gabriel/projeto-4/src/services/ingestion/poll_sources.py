from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from contracts import CompanyCode, Period, ReportType
from db import UdaRepository
from services.extractor.hash import calculate_sha256
from services.extractor.process_pdf import SRC_ROOT, load_dotenv_file, process_pdf

DEFAULT_RAW_DIR = SRC_ROOT / "data" / "raw"
DEFAULT_USER_AGENT = "uda-pipeline-projeto-4/0.1 (academic polling; daily)"
PDF_TIMEOUT_SECONDS = 30
HTML_TIMEOUT_SECONDS = 20

PERIOD_PATTERN = re.compile(r"(?P<quarter>[1-4])\s*T\s*(?:20)?(?P<year>\d{2})", re.IGNORECASE)
PREVIEW_TERMS = ("previa operacional", "prévia operacional", "operational preview")


@dataclass(frozen=True)
class PdfCandidate:
    url: str
    title: str
    period: Period | None


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def infer_period(value: str) -> Period | None:
    match = PERIOD_PATTERN.search(value)
    if match is None:
        return None
    return Period(year=2000 + int(match.group("year")), quarter=int(match.group("quarter")))


def is_operational_preview(value: str) -> bool:
    normalized = value.lower()
    return any(term in normalized for term in PREVIEW_TERMS)


def discover_pdf_links(source_url: str, *, include_all_pdfs: bool = False) -> list[PdfCandidate]:
    headers = {"User-Agent": os.getenv("UDA_USER_AGENT", DEFAULT_USER_AGENT)}
    response = requests.get(source_url, headers=headers, timeout=HTML_TIMEOUT_SECONDS)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    candidates: dict[str, PdfCandidate] = {}

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        absolute_url = urljoin(source_url, href)
        parsed = urlparse(absolute_url)
        if not parsed.path.lower().endswith(".pdf"):
            continue

        title = normalize_text(anchor.get_text(" ", strip=True) or Path(parsed.path).name)
        context = f"{title} {absolute_url}"
        if not include_all_pdfs and not is_operational_preview(context):
            continue

        period = infer_period(context)
        candidates[absolute_url] = PdfCandidate(url=absolute_url, title=title, period=period)

    return sorted(candidates.values(), key=lambda candidate: candidate.url)


def download_pdf(candidate: PdfCandidate, *, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(candidate.url).path).name or f"document_{datetime.utcnow().timestamp():.0f}.pdf"
    output_path = output_dir / filename

    headers = {"User-Agent": os.getenv("UDA_USER_AGENT", DEFAULT_USER_AGENT)}
    response = requests.get(candidate.url, headers=headers, timeout=PDF_TIMEOUT_SECONDS)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def poll_once(
    *,
    company: CompanyCode | None = None,
    include_all_pdfs: bool = False,
    process: bool = True,
) -> None:
    repo = UdaRepository()
    sources = repo.list_ingestion_sources(company_code=company.value if company else None)

    for source in sources:
        company_code = CompanyCode(source["company_code"])
        print(f"checking company={company_code.value} source={source['source_url']}")
        candidates = discover_pdf_links(source["source_url"], include_all_pdfs=include_all_pdfs)
        print(f"candidates={len(candidates)}")

        for candidate in candidates:
            if candidate.period is None:
                print(f"skip reason=missing_period url={candidate.url}")
                continue

            raw_dir = DEFAULT_RAW_DIR / company_code.value.lower() / str(candidate.period.year) / candidate.period.label.lower()
            pdf_path = download_pdf(candidate, output_dir=raw_dir)
            pdf_hash = calculate_sha256(pdf_path)

            if repo.document_exists_by_hash(pdf_hash):
                print(f"skip reason=duplicate hash={pdf_hash} url={candidate.url}")
                continue

            print(f"new_pdf hash={pdf_hash} period={candidate.period.label} url={candidate.url}")
            if process:
                process_pdf(
                    pdf_path=pdf_path,
                    company=company_code,
                    period=candidate.period,
                    source_url=candidate.url,
                    report_type=ReportType.OPERATIONAL_PREVIEW,
                )

        repo.mark_ingestion_source_checked(source["id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll RI result centers and process new operational preview PDFs.")
    parser.add_argument("--company", choices=[item.value for item in CompanyCode], default=None)
    parser.add_argument("--include-all-pdfs", action="store_true")
    parser.add_argument("--discover-only", action="store_true", help="Discover and download-check candidates without parsing.")
    return parser.parse_args()


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    poll_once(
        company=CompanyCode(args.company) if args.company else None,
        include_all_pdfs=args.include_all_pdfs,
        process=not args.discover_only,
    )


if __name__ == "__main__":
    main()
