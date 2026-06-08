from uuid import uuid4

from services.extractor.lineage import resolve_metric_chunk_id
from services.extractor.run_pipeline import materialize_pdf_input
from services.storage import MinioArtifactStorage


def test_resolve_metric_chunk_id_returns_matching_uuid() -> None:
    chunk_id = uuid4()
    chunks = [{"id": chunk_id}, {"id": uuid4()}]

    assert resolve_metric_chunk_id(str(chunk_id), chunks) == chunk_id


def test_resolve_metric_chunk_id_returns_none_for_missing_value() -> None:
    assert resolve_metric_chunk_id(None, [{"id": uuid4()}]) is None
    assert resolve_metric_chunk_id(str(uuid4()), [{"id": uuid4()}]) is None


def test_materialize_pdf_input_downloads_minio_uri(monkeypatch) -> None:
    class FakeMinioStorage(MinioArtifactStorage):
        def __init__(self) -> None:
            pass

        def get_uri_bytes(self, uri: str) -> bytes:
            assert uri == "minio://uda-artifacts/raw/file.pdf"
            return b"%PDF"

    monkeypatch.setattr("services.extractor.run_pipeline.get_artifact_storage", lambda: FakeMinioStorage())

    with materialize_pdf_input("minio://uda-artifacts/raw/file.pdf") as path:
        assert path.read_bytes() == b"%PDF"
