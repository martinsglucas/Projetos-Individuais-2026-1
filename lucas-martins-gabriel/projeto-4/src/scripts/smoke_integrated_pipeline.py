from __future__ import annotations

import argparse
import os
from pathlib import Path

import psycopg2
from minio import Minio

from services.extractor.run_pipeline import run_pipeline
from contracts import CompanyCode, Period, ReportType

SRC_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_URL = "postgresql://admin:admin@localhost:5432/uda"


def load_dotenv_file(path: Path = SRC_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def check_postgres(database_url: str) -> None:
    conn = psycopg2.connect(database_url)
    conn.close()
    print("postgres=ok")


def apply_schema(database_url: str) -> None:
    schema_path = SRC_ROOT / "db" / "001_initial_schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
    finally:
        conn.close()
    print("schema=ok")


def check_minio() -> None:
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    bucket = os.getenv("MINIO_BUCKET", "uda-artifacts")

    client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    print(f"minio=ok bucket={bucket}")


def run_fixture_pipeline() -> None:
    run_pipeline(
        pdf_path=SRC_ROOT / "data" / "raw" / "mrv_1t25.pdf",
        company=CompanyCode.MRV,
        period=Period.from_label("1T25"),
        source_url="https://ri.mrv.com.br/informacoes-financeiras/central-de-resultados/",
        report_type=ReportType.OPERATIONAL_PREVIEW,
        model_name="fixture",
        fixture_path=SRC_ROOT / "data" / "validated" / "mrv_1t25_fixture_metrics.json",
        force=True,
    )
    print("fixture_pipeline=ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Postgres, MinIO and the fixture UDA pipeline.")
    parser.add_argument("--apply-schema", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    check_postgres(database_url)
    if args.apply_schema:
        apply_schema(database_url)
    check_minio()
    if not args.skip_pipeline:
        run_fixture_pipeline()


if __name__ == "__main__":
    main()
