from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import RealDictCursor

DEFAULT_DATABASE_URL = "postgresql://admin:admin@localhost:5432/uda"


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


class Database:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or get_database_url()

    @contextmanager
    def connect(self) -> Iterator[PgConnection]:
        conn = psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
