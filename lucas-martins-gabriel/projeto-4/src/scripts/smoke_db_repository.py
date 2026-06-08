from __future__ import annotations

from datetime import datetime

from contracts import (
    CompanyCode,
    DocumentChunk,
    DocumentMetadata,
    ExtractedMetric,
    ExtractionRun,
    ExtractionStrategy,
    MetricCategory,
    MetricUnit,
    Period,
    SourceEvidence,
)
from db import UdaRepository


def main() -> None:
    pdf_hash = "b" * 64
    period = Period.from_label("1T25")
    repo = UdaRepository()

    document = DocumentMetadata(
        company=CompanyCode.MRV,
        period=period,
        source_url="https://example.com/mrv_1t25.pdf",
        pdf_hash=pdf_hash,
        storage_path="raw/MRV/2025/1T25/mrv_1t25.pdf",
        original_filename="mrv_1t25.pdf",
        discovered_at=datetime.utcnow(),
        downloaded_at=datetime.utcnow(),
    )
    document_id = repo.upsert_document(document, status="downloaded")

    chunk = DocumentChunk(
        document_hash=pdf_hash,
        ordinal=0,
        heading="DADOS OPERACIONAIS",
        page_start=4,
        page_end=4,
        content="VENDAS TOTAL INCORPORACAO VGV (R$ milhoes) 2.167",
        token_count=8,
    )
    chunk_id = repo.upsert_chunk(chunk)

    run = ExtractionRun(
        document_hash=pdf_hash,
        strategy=ExtractionStrategy.HYBRID,
        llm_provider="gemini",
        llm_model="gemini-2.5-flash",
        prompt_version="v1",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        status="succeeded",
        input_tokens=120,
        output_tokens=40,
    )
    run_id = repo.insert_extraction_run(run)

    metric = ExtractedMetric(
        company=CompanyCode.MRV,
        period=period,
        category=MetricCategory.SALES,
        metric_name="VGV",
        segment="TOTAL INCORPORACAO",
        value="2167",
        unit=MetricUnit.BRL_MILLION,
        currency="BRL",
        scale="million",
        confidence="0.95",
        evidence=SourceEvidence(
            page_number=4,
            chunk_id=str(chunk_id),
            heading="DADOS OPERACIONAIS",
            raw_text="VGV (R$ milhoes) 2.167",
        ),
    )
    metric_id = repo.insert_metric(metric, document_hash=pdf_hash, extraction_run_id=run_id, chunk_id=chunk_id)
    repo.mark_document_status(pdf_hash, "validated")

    print(f"document_id={document_id}")
    print(f"chunk_id={chunk_id}")
    print(f"run_id={run_id}")
    print(f"metric_id={metric_id}")


if __name__ == "__main__":
    main()
