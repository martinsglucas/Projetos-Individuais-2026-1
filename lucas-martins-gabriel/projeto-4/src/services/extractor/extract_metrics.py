from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from contracts import (
    CompanyCode,
    ExtractionRun,
    ExtractionStrategy,
    LLMMetricExtractionResponse,
    Period,
    ReportType,
)
from db import UdaRepository
from services.extractor.chunk_filter import select_relevant_chunks
from services.extractor.lineage import resolve_metric_chunk_id
from services.extractor.llm import GeminiProvider, parse_llm_response
from services.extractor.prompts import PROMPT_VERSION, build_metric_extraction_prompt
from services.storage import artifact_name, get_artifact_storage

SRC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESPONSE_DIR = SRC_ROOT / "data" / "validated"

def load_dotenv_file(path: Path = SRC_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def write_raw_response(document_hash: str, raw_text: str, output_dir: Path = DEFAULT_RESPONSE_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"llm_response_{document_hash[:12]}.json"
    path.write_text(raw_text, encoding="utf-8")
    return path


def relative_to_src(path: Path) -> str:
    try:
        return str(path.relative_to(SRC_ROOT))
    except ValueError:
        return str(path)


def run_extraction(
    *,
    document_hash: str,
    model_name: str,
    fixture_path: Path | None = None,
    persist_raw: bool = True,
) -> None:
    repo = UdaRepository()
    storage = get_artifact_storage()
    document = repo.get_document_by_hash(document_hash)
    chunks = repo.list_chunks(document_hash)
    period_label = f"{document['quarter']}T{str(document['year'])[-2:]}"
    selected_chunks = select_relevant_chunks(chunks, period_label=period_label)

    prompt = build_metric_extraction_prompt(
        company=document["company_code"],
        year=document["year"],
        quarter=document["quarter"],
        report_type=document["report_type"],
        chunks=selected_chunks,
    )

    started_at = datetime.utcnow()
    raw_response_path: Path | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    if fixture_path is not None:
        raw_text = fixture_path.read_text(encoding="utf-8")
        parsed = parse_llm_response(raw_text)
        provider = "fixture"
        actual_model_name = model_name
        status = "succeeded"
        finished_at = datetime.utcnow()
    else:
        provider = "gemini"
        try:
            result = GeminiProvider(model_name=model_name).extract(prompt)
        except Exception as exc:
            finished_at = datetime.utcnow()
            failed_run = ExtractionRun(
                document_hash=document_hash,
                strategy=ExtractionStrategy.HYBRID,
                parser="docling",
                llm_provider=provider,
                llm_model=model_name,
                prompt_version=PROMPT_VERSION,
                started_at=started_at,
                finished_at=finished_at,
                status="failed",
                error_message=str(exc),
            )
            repo.insert_extraction_run(failed_run)
            repo.mark_document_status(document_hash, "failed", error_message=str(exc))
            raise
        raw_text = result.raw_text
        parsed = result.parsed
        actual_model_name = result.model_name
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        status = "succeeded"
        finished_at = datetime.utcnow()

    if persist_raw:
        raw_response_path = write_raw_response(document_hash, raw_text)
        raw_response_storage_path = storage.put_text(
            text=raw_text,
            object_name=artifact_name(
                "validated",
                document["company_code"],
                str(document["year"]),
                document["period_label"],
                f"llm_response_{document_hash}.json",
            ),
            content_type="application/json",
        )
    else:
        raw_response_storage_path = None

    run = ExtractionRun(
        document_hash=document_hash,
        strategy=ExtractionStrategy.HYBRID,
        llm_provider=provider,
        llm_model=actual_model_name,
        prompt_version=PROMPT_VERSION,
        started_at=started_at,
        finished_at=finished_at,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        status=status,
    )
    run_id = repo.insert_extraction_run(
        run,
        raw_response_storage_path=raw_response_storage_path,
    )

    repo.delete_metrics_for_document(document_hash)
    metric_ids = [
        repo.insert_metric(
            metric=metric,
            document_hash=document_hash,
            extraction_run_id=run_id,
            chunk_id=resolve_metric_chunk_id(metric.evidence.chunk_id, chunks),
        )
        for metric in parsed.metrics
    ]
    repo.mark_document_status(document_hash, "extracted")

    print(f"document_hash={document_hash}")
    print(f"selected_chunks={len(selected_chunks)}")
    print(f"run_id={run_id}")
    print(f"metrics={len(metric_ids)}")
    if raw_response_path:
        print(f"raw_response={relative_to_src(raw_response_path)}")
    if raw_response_storage_path:
        print(f"raw_response_storage={raw_response_storage_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured operational metrics from parsed document chunks.")
    parser.add_argument("--pdf-hash", required=True, help="SHA-256 hash of a parsed document.")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--fixture", default=None, help="Path to a JSON fixture with LLMMetricExtractionResponse shape.")
    parser.add_argument("--no-persist-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    fixture_path = Path(args.fixture).resolve() if args.fixture else None
    run_extraction(
        document_hash=args.pdf_hash,
        model_name=args.model,
        fixture_path=fixture_path,
        persist_raw=not args.no_persist_raw,
    )


if __name__ == "__main__":
    main()
