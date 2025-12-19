from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from banditgpt._resources import get_priors_manifest_path, get_priors_path, get_user_priors_dir

logger = logging.getLogger(__name__)

_MANIFEST_CACHE: Optional["PriorsManifest"] = None


@dataclass(frozen=True)
class PriorFileInfo:
    name: str
    sha256: str
    size_bytes: int
    bundled: bool
    url: Optional[str] = None

    @property
    def bundled_path(self) -> Path:
        return get_priors_path(self.name)

    @property
    def user_path(self) -> Path:
        return get_user_priors_dir() / self.name


@dataclass(frozen=True)
class PriorsManifest:
    schema_version: str
    priors_version: str
    files: List[PriorFileInfo]

    def file_map(self) -> Dict[str, PriorFileInfo]:
        return {f.name: f for f in self.files}


class PriorIntegrityError(RuntimeError):
    """Raised when priors are missing or fail integrity checks."""


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, expected_sha256: str) -> bool:
    if not path.exists():
        return False
    try:
        return _compute_sha256(path) == expected_sha256
    except OSError:
        return False


def load_priors_manifest(manifest_path: Optional[Path] = None, cache: bool = True) -> PriorsManifest:
    global _MANIFEST_CACHE
    if cache and manifest_path is None and _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE

    path = manifest_path or get_priors_manifest_path()
    raw = json.loads(path.read_text())
    logger.debug("Loaded priors manifest", extra={"manifest_path": str(path)})
    files = [
        PriorFileInfo(
            name=entry["name"],
            sha256=entry["sha256"],
            size_bytes=int(entry["size_bytes"]),
            bundled=bool(entry.get("bundled", False)),
            url=entry.get("url"),
        )
        for entry in raw.get("files", [])
    ]
    manifest = PriorsManifest(
        schema_version=str(raw.get("schema_version", "")),
        priors_version=str(raw.get("priors_version", "")),
        files=files,
    )
    if cache and manifest_path is None:
        _MANIFEST_CACHE = manifest
    return manifest


def verify_bundled_prior(path: Path, manifest: Optional[PriorsManifest] = None) -> None:
    """
    Validate a bundled prior against the manifest checksum.

    Raises:
        PriorIntegrityError: if the file is missing, not listed, or checksum fails.
    """
    manifest = manifest or load_priors_manifest()
    entry = manifest.file_map().get(path.name)
    if entry is None:
        raise PriorIntegrityError(
            f"Priors file {path.name} is not listed in manifest (expected {manifest.priors_version}). "
            "Reinstall the package or fetch priors from git."
        )
    if not path.exists():
        raise PriorIntegrityError(
            f"Priors file missing: {path}. "
            "Reinstall the package or restore the file from git: "
            f"git show <ref>:banditgpt/data/priors/{path.name} > {path.name}"
        )
    if not validate_file(path, entry.sha256):
        raise PriorIntegrityError(
            f"Checksum mismatch for {path.name}. "
            "The file may be corrupted. Reinstall the package or replace it from git: "
            f"git show <ref>:banditgpt/data/priors/{path.name} > {path.name}"
        )
    logger.debug(
        "Verified bundled prior",
        extra={
            "prior": path.name,
            "sha256": entry.sha256,
            "size_bytes": entry.size_bytes,
            "priors_version": manifest.priors_version,
        },
    )
