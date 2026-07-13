from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

ANNEX_PATTERN = re.compile(r"SHA256E-s(?P<size>\d+)--(?P<sha>[0-9a-f]{64})(?P<suffix>\.[^/]+)")


@dataclass(frozen=True)
class AnnexFile:
    relative_path: Path
    size: int
    sha256: str


def _run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def clone_metadata(repo_url: str, destination: Path) -> Path:
    if destination.exists():
        _run("git", "-C", str(destination), "pull", "--ff-only")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", "--depth", "1", repo_url, str(destination))
    return destination


def annex_inventory(metadata_root: Path, derivatives_only: bool = True) -> list[AnnexFile]:
    inventory: list[AnnexFile] = []
    for path in sorted(metadata_root.rglob("*.set")):
        relative = path.relative_to(metadata_root)
        is_derivative = relative.parts[0] == "derivatives"
        if derivatives_only != is_derivative:
            continue
        if not path.is_symlink():
            continue
        match = ANNEX_PATTERN.search(os.readlink(path))
        if match is None:
            raise ValueError(f"Could not parse git-annex key from {path}")
        inventory.append(
            AnnexFile(
                relative_path=relative,
                size=int(match.group("size")),
                sha256=match.group("sha"),
            )
        )
    return inventory


def sha256sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download_with_resume(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > expected_size:
        destination.unlink()
        existing = 0
    headers = {"User-Agent": "rostrocaudal-dementia/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = existing > 0 and response.status == 206
        if existing and not append:
            existing = 0
        mode = "ab" if append else "wb"
        total = expected_size - existing
        with (
            destination.open(mode) as output,
            tqdm(
                total=total,
                initial=0,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as progress,
        ):
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
                progress.update(len(chunk))


def copy_metadata(metadata_root: Path, dataset_root: Path) -> None:
    for source in metadata_root.rglob("*"):
        if source.is_dir() or source.is_symlink() or ".git" in source.parts:
            continue
        relative = source.relative_to(metadata_root)
        target = dataset_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def download_dataset(
    *,
    dataset_id: str,
    repo_url: str,
    s3_prefix: str,
    data_root: Path,
    derivatives_only: bool = True,
) -> list[Path]:
    metadata_root = clone_metadata(repo_url, data_root / f"{dataset_id}-metadata")
    dataset_root = data_root / dataset_id
    dataset_root.mkdir(parents=True, exist_ok=True)
    copy_metadata(metadata_root, dataset_root)
    inventory = annex_inventory(metadata_root, derivatives_only=derivatives_only)
    downloaded: list[Path] = []
    for item in inventory:
        destination = dataset_root / item.relative_path
        if destination.exists() and destination.stat().st_size == item.size:
            if sha256sum(destination) == item.sha256:
                downloaded.append(destination)
                continue
            destination.unlink()
        url = f"{s3_prefix.rstrip('/')}/{item.relative_path.as_posix()}"
        _download_with_resume(url, destination, item.size)
        actual_hash = sha256sum(destination)
        if destination.stat().st_size != item.size or actual_hash != item.sha256:
            destination.unlink(missing_ok=True)
            raise OSError(f"Integrity check failed for {item.relative_path}")
        downloaded.append(destination)
    return downloaded


def iter_expected_set_files(dataset_root: Path) -> Iterable[Path]:
    yield from sorted((dataset_root / "derivatives").rglob("sub-*/eeg/*.set"))
