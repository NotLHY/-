"""Strict transfer manifest shared by the bundle builder and server verifier."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
            raise ValueError("Manifest paths must be portable relative paths")
        return value


class TransferManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    archive_root: str = "cxr_project"
    created_at_ms: int = Field(ge=1_000_000_000_000)
    files: list[FileEntry] = Field(min_length=1)
    note: str


class SplitCount(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    studies: int = Field(gt=0)
    images: int = Field(gt=0)


class DataManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    archive_file: Literal["cxr_first.zip"]
    archive_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    uncompressed_bytes: int = Field(gt=0)
    files: int = Field(gt=0)
    archive_root: Literal["cxr_first"]
    csv_sha256: dict[str, str]
    counts: dict[str, SplitCount]
