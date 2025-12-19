from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional

from banditgpt._resources import get_user_priors_dir
from banditgpt.core.prior_manifest import PriorsManifest, load_priors_manifest, validate_file

DEFAULT_BASE_URL_ENV = "BANDITGPT_PRIORS_BASE_URL"


class PriorDownloadError(RuntimeError):
    pass


class PriorDownloadSecurityError(PriorDownloadError):
    """Raised when download security requirements are not met."""


def _download_file(url: str, dest: Path, expected_sha256: str, timeout: float = 10.0) -> None:
    if not url.lower().startswith("https://"):
        raise PriorDownloadSecurityError(f"Refusing to download non-HTTPS URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, tmp_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:  # pragma: no cover - network errors are environmental
        raise PriorDownloadError(f"Failed to download priors from {url}: {exc}") from exc

    if not validate_file(tmp_path, expected_sha256):
        tmp_path.unlink(missing_ok=True)
        raise PriorDownloadError(f"Checksum mismatch for {url}")

    tmp_path.replace(dest)


def _resolve_base_url(manifest: PriorsManifest, override_base_url: Optional[str]) -> Optional[str]:
    if override_base_url:
        return override_base_url
    env_url = os.environ.get(DEFAULT_BASE_URL_ENV)
    if env_url:
        return env_url
    return None


def ensure_priors(
    required_files: Optional[Iterable[str]] = None,
    manifest: Optional[PriorsManifest] = None,
    base_url: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, Path]:
    """
    Ensure required priors exist locally and pass checksum validation.

    The function first checks bundled priors, then user cache (~/.banditgpt/priors).
    If a required file is missing or fails checksum, it will attempt to download
    it when a base URL is provided via argument or BANDITGPT_PRIORS_BASE_URL.
    """
    manifest = manifest or load_priors_manifest()
    file_map = manifest.file_map()
    required = list(required_files) if required_files else list(file_map)
    base_url = _resolve_base_url(manifest, base_url)

    resolved: Dict[str, Path] = {}
    for name in required:
        info = file_map.get(name)
        if info is None:
            raise PriorDownloadError(f"Unknown prior requested: {name}")

        candidates = [info.user_path, info.bundled_path]
        valid_path = next(
            (p for p in candidates if validate_file(p, info.sha256)),
            None,
        )
        if valid_path:
            resolved[name] = valid_path
            continue

        if not base_url:
            raise PriorDownloadError(
                f"Prior {name} is missing or corrupted and no base URL was provided. "
                f"Set {DEFAULT_BASE_URL_ENV} or pass base_url to enable download."
            )

        if info.url:
            url = info.url
        else:
            url = base_url.rstrip("/") + "/" + info.name

        dest = get_user_priors_dir() / info.name
        _download_file(url, dest, info.sha256, timeout=timeout)
        resolved[name] = dest

    return resolved
