from __future__ import annotations

from typing import Any
from uuid import UUID


def resolve_metric_chunk_id(metric_chunk_id: str | None, chunks: list[dict[str, Any]]) -> UUID | None:
    if not metric_chunk_id:
        return None

    chunk_ids = {str(chunk["id"]): chunk["id"] for chunk in chunks if chunk.get("id")}
    return chunk_ids.get(metric_chunk_id)
