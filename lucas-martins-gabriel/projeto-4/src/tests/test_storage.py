from services.storage.artifacts import FilesystemArtifactStorage, artifact_name


def test_filesystem_storage_writes_text(tmp_path) -> None:
    storage = FilesystemArtifactStorage(tmp_path)

    path = storage.put_text(text="ok", object_name=artifact_name("parsed", "file.md"))

    assert path.endswith("parsed/file.md")
    assert (tmp_path / "parsed" / "file.md").read_text(encoding="utf-8") == "ok"
