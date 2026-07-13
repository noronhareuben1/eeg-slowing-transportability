"""Download and verify the P-ADIC Dryad files used by the transportability study."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FileSpec:
    name: str
    file_id: int
    size: int
    sha256: str


FILES = (
    FileSpec(
        "alz_c1_new.mat",
        1891763,
        3813216803,
        "0c2fcee914d52d614596a721e385e8218017074eff9fae8a17676ba95e51576d",
    ),
    FileSpec(
        "controls_c1_new.mat",
        1891764,
        7165225913,
        "3272bb5be59f40225832d48046b85c9f79cbf343bb4328e0cf4f2d70244f5955",
    ),
    FileSpec(
        "AUTHOR_DATASET_SHOR_BENNINGER.txt",
        1891768,
        4374,
        "808b9cc79fb87df13d7fc0249b56e882edcc637de83cc1054ca055e027f64ee3",
    ),
)


def sha256(path: pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(spec: FileSpec, output: pathlib.Path) -> pathlib.Path:
    output.mkdir(parents=True, exist_ok=True)
    target = output / spec.name
    if target.exists() and target.stat().st_size == spec.size:
        if sha256(target) == spec.sha256:
            return target
        target.unlink()
    url = f"https://datadryad.org/downloads/file_stream/{spec.file_id}"
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(target), url],
        check=True,
    )
    if target.stat().st_size != spec.size or sha256(target) != spec.sha256:
        raise RuntimeError(f"checksum or size mismatch for {spec.name}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify files already downloaded through a browser; do not contact Dryad",
    )
    args = parser.parse_args()
    records = []
    for spec in FILES:
        path = args.output / spec.name
        if args.verify_only:
            if not path.exists():
                raise FileNotFoundError(path)
            if path.stat().st_size != spec.size or sha256(path) != spec.sha256:
                raise RuntimeError(f"checksum or size mismatch for {spec.name}")
        else:
            path = download(spec, args.output)
        records.append({**asdict(spec), "path": str(path)})
    manifest = {
        "doi": "10.5061/dryad.8gtht76pw",
        "version_date": "2022-10-27",
        "files": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
