from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_manifest(
    path: Path,
    *,
    stage: str,
    project_root: Path,
    config_path: Path,
    inputs: list[Path],
    outputs: list[Path],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "stage": stage,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(project_root),
        "config_sha256": file_sha256(config_path),
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {str(item): file_sha256(item) for item in inputs if item.is_file()},
        "outputs": {str(item): file_sha256(item) for item in outputs if item.is_file()},
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
