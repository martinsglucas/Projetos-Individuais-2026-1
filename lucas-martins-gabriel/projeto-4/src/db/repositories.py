from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from psycopg2.extras import Json

from contracts import DocumentChunk, DocumentMetadata, ExtractedMetric, ExtractionRun
from db.connection import Database


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _jsonb(value: dict[str, Any]) -> Json:
    return Json(value, dumps=json.dumps)


class UdaRepository:
    def __init__(self, database: Database | None = None) -> None:
        self.database = database or Database()

    def get_company_id(self, company_code: str) -> UUID:
        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM companies WHERE code = %s", (company_code,))
            row = cur.fetchone()
            if row is None:
                raise LookupError(f"company not found: {company_code}")
            return row["id"]

    def list_companies(self) -> list[dict[str, Any]]:
        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT code, name, ri_base_url
                FROM companies
                ORDER BY code
                """
            )
            return [dict(row) for row in cur.fetchall()]

    def list_ingestion_sources(self, company_code: str | None = None) -> list[dict[str, Any]]:
        filters = ["s.polling_enabled = true"]
        params: list[Any] = []

        if company_code:
            filters.append("c.code = %s")
            params.append(company_code)

        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    s.id,
                    c.code AS company_code,
                    s.source_name,
                    s.source_url,
                    s.last_checked_at,
                    s.metadata
                FROM ingestion_sources s
                JOIN companies c ON c.id = s.company_id
                WHERE {' AND '.join(filters)}
                ORDER BY c.code, s.source_name
                """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]

    def mark_ingestion_source_checked(self, source_id: UUID) -> None:
        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE ingestion_sources SET last_checked_at = %s WHERE id = %s", (datetime.utcnow(), source_id))

    def get_document_by_hash(self, pdf_hash: str) -> dict[str, Any]:
        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    d.id,
                    c.code AS company_code,
                    d.report_type,
                    d.year,
                    d.quarter,
                    d.period_label,
                    d.source_url,
                    d.pdf_hash,
                    d.status,
                    d.storage_path,
                    d.parsed_storage_path
                FROM documents d
                JOIN companies c ON c.id = d.company_id
                WHERE d.pdf_hash = %s
                """,
                (pdf_hash,),
            )
            row = cur.fetchone()
            if row is None:
                raise LookupError(f"document not found for hash: {pdf_hash}")
            return dict(row)

    def document_exists_by_hash(self, pdf_hash: str) -> bool:
        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM documents WHERE pdf_hash = %s", (pdf_hash,))
            return cur.fetchone() is not None

    def list_documents(
        self,
        *,
        company_code: str | None = None,
        year: int | None = None,
        quarter: int | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[str] = []
        params: list[Any] = []

        if company_code:
            filters.append("c.code = %s")
            params.append(company_code)
        if year:
            filters.append("d.year = %s")
            params.append(year)
        if quarter:
            filters.append("d.quarter = %s")
            params.append(quarter)

        where = f"WHERE {' AND '.join(filters)}" if filters else ""

        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    d.id,
                    c.code AS company_code,
                    d.report_type,
                    d.year,
                    d.quarter,
                    d.period_label,
                    d.source_url,
                    d.pdf_hash,
                    d.original_filename,
                    d.storage_path,
                    d.parsed_storage_path,
                    d.validated_storage_path,
                    d.status,
                    d.discovered_at,
                    d.downloaded_at,
                    d.parsed_at,
                    d.processed_at,
                    d.error_message
                FROM documents d
                JOIN companies c ON c.id = d.company_id
                {where}
                ORDER BY d.year DESC, d.quarter DESC, c.code
                """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]

    def list_chunks(self, pdf_hash: str) -> list[dict[str, Any]]:
        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ch.id,
                    ch.ordinal,
                    ch.heading,
                    ch.page_start,
                    ch.page_end,
                    ch.content,
                    ch.token_count,
                    ch.metadata
                FROM document_chunks ch
                JOIN documents d ON d.id = ch.document_id
                WHERE d.pdf_hash = %s
                ORDER BY ch.ordinal
                """,
                (pdf_hash,),
            )
            return [dict(row) for row in cur.fetchall()]

    def list_metrics(
        self,
        *,
        company_code: str | None = None,
        year: int | None = None,
        quarter: int | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[str] = []
        params: list[Any] = []

        if company_code:
            filters.append("c.code = %s")
            params.append(company_code)
        if year:
            filters.append("m.year = %s")
            params.append(year)
        if quarter:
            filters.append("m.quarter = %s")
            params.append(quarter)
        if category:
            filters.append("m.category = %s")
            params.append(category)

        where = f"WHERE {' AND '.join(filters)}" if filters else ""

        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    m.id,
                    c.code AS company_code,
                    m.year,
                    m.quarter,
                    m.category,
                    m.metric_name,
                    m.segment,
                    m.value,
                    m.unit,
                    m.currency,
                    m.scale,
                    m.is_estimated,
                    m.confidence,
                    m.source_page,
                    m.source_heading,
                    m.source_text,
                    m.table_label,
                    d.pdf_hash,
                    d.source_url,
                    d.storage_path,
                    m.extracted_at
                FROM metrics m
                JOIN companies c ON c.id = m.company_id
                JOIN documents d ON d.id = m.document_id
                {where}
                ORDER BY m.year DESC, m.quarter DESC, c.code, m.category, m.metric_name
                """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]

    def list_conjuntura_metric_rows(
        self,
        *,
        year: int,
        quarter: int,
        company_code: str | None = None,
    ) -> list[dict[str, Any]]:
        start_year = year - 2
        categories = ("launches", "sales")
        filters = [
            "m.year BETWEEN %s AND %s",
            "m.quarter BETWEEN 1 AND %s",
            "m.category::text = ANY(%s)",
            "m.metric_name IN ('VGV', 'VGV acumulado')",
            "(m.segment IS NULL OR upper(m.segment) IN ('TOTAL INCORPORACAO', 'TOTAL INCORPORAÇÃO', 'TOTAL'))",
        ]
        params: list[Any] = [start_year, year, quarter, list(categories)]

        if company_code:
            filters.append("c.code = %s")
            params.append(company_code)

        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    c.code AS company_code,
                    m.year,
                    m.quarter,
                    m.category,
                    m.metric_name,
                    m.segment,
                    m.value,
                    m.unit,
                    m.currency,
                    m.scale,
                    m.source_page,
                    m.source_text,
                    d.pdf_hash,
                    d.source_url
                FROM metrics m
                JOIN companies c ON c.id = m.company_id
                JOIN documents d ON d.id = m.document_id
                WHERE {' AND '.join(filters)}
                ORDER BY c.code, m.category, m.year, m.quarter
                """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]

    def upsert_document(self, document: DocumentMetadata, status: str = "downloaded") -> UUID:
        company_code = _enum_value(document.company)
        report_type = _enum_value(document.report_type)

        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM companies WHERE code = %s", (company_code,))
            company = cur.fetchone()
            if company is None:
                raise LookupError(f"company not found: {company_code}")

            cur.execute(
                """
                INSERT INTO documents (
                    company_id, report_type, year, quarter, source_url, pdf_hash,
                    original_filename, storage_path, status, discovered_at, downloaded_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pdf_hash) DO UPDATE SET
                    source_url = COALESCE(EXCLUDED.source_url, documents.source_url),
                    original_filename = COALESCE(EXCLUDED.original_filename, documents.original_filename),
                    storage_path = COALESCE(EXCLUDED.storage_path, documents.storage_path),
                    status = EXCLUDED.status,
                    downloaded_at = COALESCE(EXCLUDED.downloaded_at, documents.downloaded_at)
                RETURNING id
                """,
                (
                    company["id"],
                    report_type,
                    document.period.year,
                    document.period.quarter,
                    document.source_url,
                    document.pdf_hash,
                    document.original_filename,
                    document.storage_path,
                    status,
                    document.discovered_at,
                    document.downloaded_at,
                ),
            )
            return cur.fetchone()["id"]

    def mark_document_status(self, pdf_hash: str, status: str, error_message: str | None = None) -> None:
        timestamp_column = {
            "parsed": "parsed_at",
            "extracted": "processed_at",
            "validated": "processed_at",
        }.get(status)

        with self.database.connect() as conn, conn.cursor() as cur:
            if timestamp_column:
                cur.execute(
                    f"UPDATE documents SET status = %s, error_message = %s, {timestamp_column} = %s WHERE pdf_hash = %s",
                    (status, error_message, datetime.utcnow(), pdf_hash),
                )
            else:
                cur.execute(
                    "UPDATE documents SET status = %s, error_message = %s WHERE pdf_hash = %s",
                    (status, error_message, pdf_hash),
                )

    def mark_document_parsed(self, pdf_hash: str, parsed_storage_path: str) -> None:
        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                SET status = 'parsed', parsed_storage_path = %s, parsed_at = %s, error_message = NULL
                WHERE pdf_hash = %s
                """,
                (parsed_storage_path, datetime.utcnow(), pdf_hash),
            )


    def insert_extraction_run(self, run: ExtractionRun, raw_response_storage_path: str | None = None) -> UUID:
        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE pdf_hash = %s", (run.document_hash,))
            document = cur.fetchone()
            if document is None:
                raise LookupError(f"document not found for hash: {run.document_hash}")

            cur.execute(
                """
                INSERT INTO extraction_runs (
                    document_id, strategy, parser, llm_provider, llm_model, prompt_version,
                    status, input_tokens, output_tokens, raw_response_storage_path,
                    error_message, started_at, finished_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    document["id"],
                    _enum_value(run.strategy),
                    run.parser,
                    run.llm_provider,
                    run.llm_model,
                    run.prompt_version,
                    run.status,
                    run.input_tokens,
                    run.output_tokens,
                    raw_response_storage_path,
                    run.error_message,
                    run.started_at,
                    run.finished_at,
                ),
            )
            return cur.fetchone()["id"]

    def upsert_chunk(self, chunk: DocumentChunk) -> UUID:
        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE pdf_hash = %s", (chunk.document_hash,))
            document = cur.fetchone()
            if document is None:
                raise LookupError(f"document not found for hash: {chunk.document_hash}")

            cur.execute(
                """
                INSERT INTO document_chunks (
                    document_id, ordinal, heading, page_start, page_end, content,
                    token_count, parser, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id, ordinal) DO UPDATE SET
                    heading = EXCLUDED.heading,
                    page_start = EXCLUDED.page_start,
                    page_end = EXCLUDED.page_end,
                    content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    parser = EXCLUDED.parser,
                    metadata = EXCLUDED.metadata
                RETURNING id
                """,
                (
                    document["id"],
                    chunk.ordinal,
                    chunk.heading,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.content,
                    chunk.token_count,
                    chunk.parser,
                    _jsonb(chunk.metadata),
                ),
            )
            return cur.fetchone()["id"]

    def replace_chunks(self, document_hash: str, chunks: list[DocumentChunk]) -> list[UUID]:
        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE pdf_hash = %s", (document_hash,))
            document = cur.fetchone()
            if document is None:
                raise LookupError(f"document not found for hash: {document_hash}")

            document_id = document["id"]
            cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))

            chunk_ids: list[UUID] = []
            for chunk in chunks:
                cur.execute(
                    """
                    INSERT INTO document_chunks (
                        document_id, ordinal, heading, page_start, page_end, content,
                        token_count, parser, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        document_id,
                        chunk.ordinal,
                        chunk.heading,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.content,
                        chunk.token_count,
                        chunk.parser,
                        _jsonb(chunk.metadata),
                    ),
                )
                chunk_ids.append(cur.fetchone()["id"])

            return chunk_ids


    def delete_metrics_for_document(self, document_hash: str) -> int:
        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM metrics m
                USING documents d
                WHERE m.document_id = d.id AND d.pdf_hash = %s
                """,
                (document_hash,),
            )
            return cur.rowcount

    def insert_metric(
        self,
        metric: ExtractedMetric,
        document_hash: str,
        extraction_run_id: UUID | None = None,
        chunk_id: UUID | None = None,
    ) -> UUID:
        company_code = _enum_value(metric.company)

        with self.database.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.id AS document_id, c.id AS company_id
                FROM documents d
                JOIN companies c ON c.id = d.company_id
                WHERE d.pdf_hash = %s AND c.code = %s
                """,
                (document_hash, company_code),
            )
            row = cur.fetchone()
            if row is None:
                raise LookupError(f"document not found for hash/company: {document_hash}/{company_code}")

            cur.execute(
                """
                INSERT INTO metrics (
                    document_id, extraction_run_id, chunk_id, company_id, year, quarter,
                    category, metric_name, segment, value, unit, currency, scale,
                    is_estimated, confidence, source_page, source_heading, source_text, table_label
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    row["document_id"],
                    extraction_run_id,
                    chunk_id,
                    row["company_id"],
                    metric.period.year,
                    metric.period.quarter,
                    _enum_value(metric.category),
                    metric.metric_name,
                    metric.segment,
                    metric.value,
                    _enum_value(metric.unit) if metric.unit else None,
                    metric.currency,
                    metric.scale,
                    metric.is_estimated,
                    metric.confidence,
                    metric.evidence.page_number,
                    metric.evidence.heading,
                    metric.evidence.raw_text,
                    metric.evidence.table_label,
                ),
            )
            return cur.fetchone()["id"]

    def insert_metric_batch(
        self,
        metrics: list[ExtractedMetric],
        document_hash: str,
        extraction_run_id: UUID | None = None,
        chunk_id: UUID | None = None,
    ) -> list[UUID]:
        return [
            self.insert_metric(
                metric=metric,
                document_hash=document_hash,
                extraction_run_id=extraction_run_id,
                chunk_id=chunk_id,
            )
            for metric in metrics
        ]
