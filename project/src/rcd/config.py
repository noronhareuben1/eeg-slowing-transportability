from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    outputs: Path
    config: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> ProjectPaths:
        current = (start or Path.cwd()).resolve()
        for candidate in (current, *current.parents):
            config = candidate / "configs" / "study.yaml"
            if config.exists():
                return cls(
                    root=candidate,
                    data=candidate / "data",
                    outputs=candidate / "outputs",
                    config=config,
                )
        raise FileNotFoundError("Could not find configs/study.yaml in this directory or a parent")


def load_config(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = ProjectPaths.discover().config
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return config
