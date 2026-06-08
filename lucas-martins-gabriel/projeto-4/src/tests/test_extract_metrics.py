from uuid import uuid4

from services.extractor.lineage import resolve_metric_chunk_id


def test_resolve_metric_chunk_id_returns_matching_uuid() -> None:
    chunk_id = uuid4()
    chunks = [{"id": chunk_id}, {"id": uuid4()}]

    assert resolve_metric_chunk_id(str(chunk_id), chunks) == chunk_id


def test_resolve_metric_chunk_id_returns_none_for_missing_value() -> None:
    assert resolve_metric_chunk_id(None, [{"id": uuid4()}]) is None
    assert resolve_metric_chunk_id(str(uuid4()), [{"id": uuid4()}]) is None
