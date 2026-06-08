import pytest

from services.storage.artifacts import FilesystemArtifactStorage, artifact_name, parse_minio_uri


def test_filesystem_storage_writes_text(tmp_path) -> None:
    storage = FilesystemArtifactStorage(tmp_path)

    path = storage.put_text(text="ok", object_name=artifact_name("parsed", "file.md"))

    assert path.endswith("parsed/file.md")
    assert (tmp_path / "parsed" / "file.md").read_text(encoding="utf-8") == "ok"


def test_filesystem_storage_reads_bytes(tmp_path) -> None:
    storage = FilesystemArtifactStorage(tmp_path)
    storage.put_text(text="ok", object_name="parsed/file.md")

    assert storage.get_bytes(object_name="parsed/file.md") == b"ok"


def test_parse_minio_uri() -> None:
    assert parse_minio_uri("minio://uda-artifacts/raw/MRV/file.pdf") == (
        "uda-artifacts",
        "raw/MRV/file.pdf",
    )


def test_parse_minio_uri_rejects_invalid_uri() -> None:
    with pytest.raises(ValueError):
        parse_minio_uri("raw/MRV/file.pdf")
