from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse


SRC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_ARTIFACT_DIR = SRC_ROOT / "data" / "artifacts"


class ArtifactStorage(ABC):
    @abstractmethod
    def put_bytes(self, *, data: bytes, object_name: str, content_type: str | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_bytes(self, *, object_name: str) -> bytes:
        raise NotImplementedError

    def put_text(self, *, text: str, object_name: str, content_type: str = "text/plain; charset=utf-8") -> str:
        return self.put_bytes(data=text.encode("utf-8"), object_name=object_name, content_type=content_type)

    def put_file(self, *, path: str | Path, object_name: str, content_type: str | None = None) -> str:
        return self.put_bytes(data=Path(path).read_bytes(), object_name=object_name, content_type=content_type)


class FilesystemArtifactStorage(ArtifactStorage):
    def __init__(self, base_dir: str | Path = DEFAULT_LOCAL_ARTIFACT_DIR) -> None:
        self.base_dir = Path(base_dir)

    def put_bytes(self, *, data: bytes, object_name: str, content_type: str | None = None) -> str:
        del content_type
        output_path = self.base_dir / object_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        try:
            return str(output_path.relative_to(SRC_ROOT))
        except ValueError:
            return str(output_path)

    def get_bytes(self, *, object_name: str) -> bytes:
        return (self.base_dir / object_name).read_bytes()


class MinioArtifactStorage(ArtifactStorage):
    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = False,
    ) -> None:
        from minio import Minio
        from minio.error import S3Error

        self.bucket_name = bucket_name
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
        except S3Error as exc:
            raise RuntimeError(f"failed to initialize MinIO bucket {bucket_name}: {exc}") from exc

    def put_bytes(self, *, data: bytes, object_name: str, content_type: str | None = None) -> str:
        from io import BytesIO

        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type or "application/octet-stream",
        )
        return f"minio://{self.bucket_name}/{object_name}"

    def get_bytes(self, *, object_name: str) -> bytes:
        response = self.client.get_object(self.bucket_name, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def get_uri_bytes(self, uri: str) -> bytes:
        bucket_name, object_name = parse_minio_uri(uri)
        response = self.client.get_object(bucket_name, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()


def get_artifact_storage() -> ArtifactStorage:
    backend = os.getenv("ARTIFACT_STORAGE_BACKEND", "minio").lower()
    if backend == "filesystem":
        return FilesystemArtifactStorage(os.getenv("ARTIFACT_LOCAL_DIR", str(DEFAULT_LOCAL_ARTIFACT_DIR)))

    return MinioArtifactStorage(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        bucket_name=os.getenv("MINIO_BUCKET", "uda-artifacts"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )


def artifact_name(*parts: str) -> str:
    cleaned = [part.strip("/").replace(" ", "_") for part in parts if part]
    return "/".join(cleaned)


def parse_minio_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "minio" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("MinIO URI must use format minio://bucket/object")
    return parsed.netloc, parsed.path.lstrip("/")
